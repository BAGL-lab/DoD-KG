import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
#FyY9tHKpnaW4RFHrb0N7x6rdfQ0d6CU1R5NgpKe4gdw=
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import os

datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = './data',
        output_dir: str = './lora_fed_dod_7b/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 64,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 10,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = data_path+'/'+str(num_clients)
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=50,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        #global_evaluation()
        #"Administrative processing, gear issue, and haircuts mark the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI.", "category": "Department of Defense"}, {"instruction": "What initiative is the DoD pursuing regarding the supply chain for rare earth materials?", "context": "", "response": "The DoD is looking to establish a 'Mine-to-Magnet' supply chain for rare earth materials.", "category": "Department of Defense"}, {"instruction": "What is the significance of Daylight Savings as mentioned in the tweet?", "context": "", "response": "Daylight Savings signifies setting clocks ahead to embrace an extra hour of daylight.", "category": "Department of Defense"}, {"instruction": "What historical event is commemorated annually as Flag Day in the U.S.?", "context": "", "response": "Flag Day is commemorated annually in the U.S. to mark the organization of the Continental Army on June 14, 1775.", "category": "Department of Defense"}, {"instruction": "What details were revealed about USAF Chief Master Sgt. Richard Etchberger's heroics during the Vietnam War?", "context": "", "response": "The declassified details revealed that USAF Chief Master Sgt. Richard Etchberger gave his life during the Vietnam War and was posthumously awarded the Medal of Honor 42 years later.", "category": "Department of Defense"}, {"instruction": "What event showcased the airpower of the USMC and USAF at MCAS Yuma, Arizona?", "context": "", "response": "The annual Yuma Airshow at MCAS Yuma, Arizona showcased the airpower of the USMC and USAF.", "category": "Department of Defense"}, {"instruction": "What humanitarian effort is the DoD undertaking in Gaza?", "context": "", "response": "The DoD is constructing a pier to deliver humanitarian aid to Gaza.", "category": "Department of Defense"}, {"instruction": "Who is Gladys West and what is her contribution to the DoD?", "context": "", "response": "Gladys West is a trailblazer who played a crucial role in U.S. military computing during the Cold War. Her work laid the foundation for the Global Positioning System (GPS).", "category": "Department of Defense"}, {"instruction": "What were the topics of discussion at the meeting of Western Hemisphere leaders?", "context": "", "response": "The leaders discussed security priorities for the Western Hemisphere.", "category": "Department of Defense"}, {"instruction": "What significant change did the National Security Act of 1947 introduce?", "context": "", "response": "The National Security Act of 1947 created a unified military command known as the National Military Establishment, which was later renamed the Department of Defense, and it also established the Central Intelligence Agency, the National Security Council, and the United States Air Force.", "category": "Department of Defense"}, {"instruction": "What was the main mission of the Office of Homeland Security?", "context": "", "response": "The main mission of the Office of Homeland Security was to develop and coordinate the implementation of a comprehensive national strategy to secure the United States from terrorist threats or attacks.", "category": "Department of Homeland Security"}, {"instruction": "How many employees does the Department of Homeland Security (DHS) have?", "context": "", "response": "The Department of Homeland Security (DHS) has more than 240,000 employees.", "category": "Department of Homeland Security"}, {"instruction": "What action did DHS take regarding immigration raids at job sites in 2021?", "context": "", "response": "In 2021, DHS halted large-scale immigration raids at job sites, planning a new enforcement strategy to more effectively target employers who pay substandard wages and engage in exploitative labor practices.", "category": "Department of Homeland Security"}, {"instruction": "When did the Department of Homeland Security (DHS) begin operations?", "context": "", "response": "The Department of Homeland Security (DHS) began operations on March 1, 2003.", "category": "Department of Homeland Security"}, {"instruction": "What is the main responsibility of the United States Department of Homeland Security (DHS)?", "context": "", "response": "The main responsibility of the United States Department of Homeland Security (DHS) is public security, which includes missions such as anti-terrorism, border security, immigration and customs, cyber security, and disaster prevention and management.", "category": "Department of Homeland Security"}, {"instruction": "Which former judge was confirmed as the Secretary of Homeland Security in 2005?", "context": "", "response": "Federal judge Michael Chertoff was confirmed as the Secretary of Homeland Security in 2005.", "category": "Department of Homeland Security"}, {"instruction": "Which agencies have significant homeland security responsibilities apart from the DHS?", "context": "", "response": "Other agencies with significant homeland security responsibilities include the Departments of Health and Human Services, Justice, and Energy.", "category": "Department of Homeland Security"}, {"instruction": "Why was the Department of Homeland Security (DHS) formed?", "context": "", "response": "The Department of Homeland Security (DHS) was formed as a result of the Homeland Security Act of 2002, enacted in response to the September 11 attacks.", "category": "Department of Homeland Security"}, {"instruction": "What was the purpose of the Cybersecurity and Infrastructure Security Agency Act of 2018?", "context": "", "response": "The purpose of the Cybersecurity and Infrastructure Security Agency Act of 2018 was to elevate the mission of the former DHS National Protection and Programs Directorate and establish the Cybersecurity and Infrastructure Security Agency.", "category": "Department of Homeland Security"}, {"instruction": "Who was the first director of the Office of Homeland Security?", "context": "", "response": "The first director of the Office of Homeland Security was former Pennsylvania Governor Tom Ridge.", "category": "Department of Homeland Security"}, {"instruction": "When was NASA established?", "context": "", "response": "NASA was established in 1958.", "category": "NASA"}, {"instruction": "What agency did NASA succeed?", "context": "", "response": "NASA succeeded the National Advisory Committee for Aeronautics (NACA).", "category": "NASA"}, {"instruction": "When did NASA begin operations?", "context": "", "response": "NASA began operations on October 1, 1958.", "category": "NASA"}, {"instruction": "What led to the creation of NACA?", "context": "", "response": "The United States created NACA in 1915 after recognizing it was far behind Europe in aviation capability, with the goal of fostering aeronautical research and development.", "category": "NASA"}, {"instruction": "What current programs does NASA support?", "context": "", "response": "NASA currently supports the International Space Station, the development of the Orion spacecraft, the Space Launch System for the crewed lunar Artemis program, the Commercial Crew spacecraft, and the planned Lunar Gateway space station.", "category": "NASA"}, {"instruction": "What are some of NASA's major space exploration programs?", "context": "", "response": "Some of NASA's major space exploration programs include Project Mercury, Project Gemini, the Apollo Moon landing missions, the Skylab space station, and the Space Shuttle.", "category": "NASA"}, {"instruction": "What does the Launch Services Program oversee?", "context": "", "response": "The Launch Services Program oversees launch operations and countdown management for NASA's uncrewed launches.", "category": "NASA"}, {"instruction": "What event ushered in the Space Age and kicked off the Space Race?", "context": "", "response": "The Soviet Union's launch of Sputnik 1 ushered in the Space Age and kicked off the Space Race.", "category": "NASA"}, {"instruction": "What is the focus of NASA's science programs?", "context": "", "response": "NASA's science programs focus on better understanding Earth through the Earth Observing System, advancing heliophysics through the Science Mission Directorate's Heliophysics Research Program, exploring bodies throughout the Solar System with robotic spacecraft like New Horizons and rovers like Perseverance, and researching astrophysics topics like the Big Bang through the James Webb Space Telescope and the Great Observatories.", "category": "NASA"}, {"instruction": "Who were the Mercury 7 astronauts selected from?", "context": "", "response": "The Mercury 7 astronauts were selected from the military, including three Air Force pilots, three Navy aviators, and one Marine Corps pilot."
        q1="What is the budget request for FY25 for the Department of the Air Force?"
        a1="The Department of the Air Force unveiled a $217.5B budget request for FY25 designed to continue modernizing the Air Force and Space Force, maintain readiness, and address key capability gaps while investing to manage risks that are increasing with time."
        q2='What activity did the Marines from the 9th Engineer Support Battalion engage in at Warrior Shield 24?'
        a2='The Marines from the 9th Engineer Support Battalion engaged targets like pros at Mohican Range, Rodriguez Live Fire Complex in South Korea during Warrior Shield 24.'
        q3='What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?'
        a3='The U.S., Australian, and British Land Chiefs addressed the progress of their security agreement.'
        q4='What are the main focuses of the 2025 DoD budget spotlight?'
        a4='The 2025 DoD budget spotlight focuses on fortifying national defense, deterring aggression, and crafting a modern defense ecosystem for 21st-century challenges.'
        q5='What does the story of the Harlem Hellfighters celebrate?'
        a5="The story of the Harlem Hellfighters celebrates their courage and triumph, which continues to inspire and live on."
        
        
        model_inputs = tokenizer([q1,q2,q3,q4,q5], return_tensors="pt",padding=True)
        generated_ids = model.generate(**model_inputs,max_length= 60, do_sample=True, top_p=0.84, top_k=100)
        res=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print('@@@',res)

            




if __name__ == "__main__":
    fire.Fire(fl_finetune)
