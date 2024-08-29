#FyY9tHKpnaW4RFHrb0N7x6rdfQ0d6CU1R5NgpKe4gdw=
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import os

q1="What is the budget request for FY25 for the Department of the Air Force?"
a1="The Department of the Air Force unveiled a $217.5B budget request for FY25 designed to continue modernizing the Air Force and Space Force, maintain readiness, and address key capability gaps while investing to manage risks that are increasing with time."
q2='What activity did the Marines from the 9th Engineer Support Battalion engage in at Warrior Shield 24?'
a2='The Marines from the 9th Engineer Support Battalion engaged targets like pros at Mohican Range, Rodriguez Live Fire Complex in South Korea during Warrior Shield 24.'
        
r1=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany\nRP Fresh\nCompany R?P\nFresh? Company R?P?\nCompany? P?resh Company\nP?Company? Company?", 'What are the main focuses of the 2025 DoD budget spotlight?\nFocused onD2\nD\nWhere2 focus Do Where the budget2D on What are the2 budget\nF Where2 the Do\nFocus budget2', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs? Did British and Aborigines, Whites and Coloreds Blacks, Wizards White and Wise.\nBand BritishandWhigand British Whiteand', 'What are the main focuses of the 2025 DoD budget spotlight? The focus of the The focus the budget the light? 2 5 the budget The2 light of budget? The5 The 2 of budget 2 The5 budget', 'What does the story of the Harlem Hellfighters celebrate?\nThe story Harlem celebrate Hell? story? celebrate? Harlem story?\nWhat does celebrate? story? Harlems what?\nstory? Har']
r2=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany\nRP Fresh\nCompany DP\nCompany R\nP D Company R Company P D Company R Company P Company D Company R Company P Company D Company", 'What are the main focuses of the 2025 DoD budget spotlight?\nFocused Attention2\nThe 02 What2 is the2 0 Attention2 Focus 2 0 2 The2\nAttention', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?\nAboriginal between bothers? The was discussed? British and Australian, between and Australian? British, bothers British between British bothers? and Australian Britishers both', 'What are the main focuses of the 2025 DoD budget spotlight? The focus of the The focus the budget the light focus the light the focus the focus the light the focus the budget The focus the light the focus the budget The focus the budget', 'What does the story of the Harlem Hellfighters celebrate?\nThere celebrates celebrating celebrates celebrate? celebrating celebrates celebrates celebrates celebrates celebrating celebrates celebrates celebrates celebrates celebrates celebrates']
r3=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany\nRP Fresh\nCompany DP\nCompany D\nP D Company DP\nP Company D\nP D Company\nP D\nP Company\n", 'What are the main focuses of the 2025 DoD budget spotlight?\nFocused onD offlights\n1 offD1 onfocus Do1 focus onD2 off1 off\nWhat onD focus Do\n1 budgetOff', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?\nAboriginal the and Government the First World,? the\nBritish and German, and\nFrance and\nBetween France and Dutch\nGerman and', 'What are the main focuses of the 2025 DoD budget spotlight? The focus of the light0 the light5 light2 5 5 light the budget2 light 0 the light5 of 2 light budget5 0 light2', 'What does the story of the Harlem Hellfighters celebrate?\nA story Harlem celebrate Hell? story? celebrate? Harlem story?\nFigure? story? Figure? story?\nFigure? story?Fig']
r4=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany\nRP Fresh\nCompany D?\nCompany?\n? D Company? Company? D? Company? Company? Company? Company? Company? Company?", 'What are the main focuses of the 2025 DoD budget spotlight?\nFocused Attention2\nThe 3Focused on theF 4budget2 Focus 4 Focus 4 the\n4 budget ', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?\nAboriginal the and Government the First World, Nation the , the of Nations.\nthe.\nthe.the\n.the\n.the\n.the\n', 'What are the main focuses of the 2025 DoD budget spotlight? The focus of the light? The light of the focus the light of the focus the light the focus the focus the light the focus the focus the focus the focus the focus the', 'What does the story of the Harlem Hellfighters celebrate?\nThere celebrates?']
r5=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI? WhatPID? WhatPID? WhatIC? WhatIDP? WhatID? WhatIC? WhatID? WhatID? WhatIC? WhatID? WhatID", 'What are the main focuses of the 2025 DoD budget spotlight?\nYou focus1 budget2 budget3\n5 budget3 main2 light4 budget2 focus2 focus2 light3\n3 focus2 main3 budget2 focus1 light', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs? Why? Meeting? Why Australian and British Lands? What British and Australian Why and Why Australia and British What and Why Australian and Why? British and What and Why Australia', 'What are the main focuses of the 2025 DoD budget spotlight?\nThe focus of the? The? the? The? the? the? The? the? The? the? The? the? the? the? The? the', 'What does the story of the Harlem Hellfighters celebrate?\nThe c’elarate? 11e1 ?1 el1 m1 ?1el1 m1 el1 m1e1 ?1 1']
r6=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany\nRP Fresh\nCompany D?\nCompany?\n? D Company? Company? D? Company? Company? Company? Company? Company? Company?", 'What are the main focuses of the 2025 DoD budget spotlight?\nFocused Attention2\nThe 3Focused on theF 4budget2 Focus 4 Focus 4 the\n4 budget ', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?\nAboriginal the and Government the First World, Nation the , the of Nations.\nthe.\nthe.the\n.the\n.the\n.the\n', 'What are the main focuses of the 2025 DoD budget spotlight? The focus of the light? The light of the focus the light of the focus the light the focus the focus the light the focus the focus the focus the focus the focus the', 'What does the story of the Harlem Hellfighters celebrate?\nThere celebrates?']
r7=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany\nRP?Company\nRCompany?P\n?Company?R?Company?P\nThe?Company?R?Company?P?Company?R", 'What are the main focuses of the 2025 DoD budget spotlight?\nFocused Attention2\nThe 3Highlight1Focus2 High3Att2ion2 Focus3\n4Focus2 High3\n4Attention', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?\nAboriginal the and Government the First , the First the , the First Nations the First the First\nThe First the\nThe First the First the First', 'What are the main focuses of the 2025 DoD budget spotlight? The focus of the light? The light of the focus the light of the focus the focus the focus the focus the focus the focus the focus the focus the focus the focus the', 'What does the story of the Harlem Hellfighters celebrate?\nA story? celebrate? Harlem story? celebrate? Harlem Story?\nFigure? Story? Figure\nFigure? Figure?\nFigure?']
r8=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI? WhatPID? WhatPID? WhatIC? WhatIDP? WhatID? WhatIC? WhatID? WhatID? WhatIC? WhatID? WhatID", 'What are the main focuses of the 2025 DoD budget spotlight?\nYou focus1 budget2 budget3\n5 budget3 main2 light4 budget2 focus2 focus2 light3\n3 focus2 main3 budget2 focus1 light', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs? Why? Meeting? Why Australian and British Lands? What British and Australian Why and Why Australia and British What and Why Australian and Why? British and What and Why Australia', 'What are the main focuses of the 2025 DoD budget spotlight?\nThe focus of the? The? the? The? the? the? The? the? The? the? The? the? the? the? The? the', 'What does the story of the Harlem Hellfighters celebrate?\nThe c’elarate? 11e1 ?1 el1 m1 ?1el1 m1 el1 m1e1 ?1 1']
r9=["What marks the beginning of the transformative journey for Bravo Company's fresh recruits at MCRDPI?\nCompany's F's'D?\nWhat's Ms'\n's?\ns's\ns'D's's?s", 'What are the main focuses of the 2025 DoD budget spotlight?\nThe spotlight? The?2 of2? The?2 spot2', 'What was discussed during the meeting between the U.S., Australian, and British Land Chiefs?\nGeographical Names British?\n(British/Britain and Brunei\nThe? British ish andh andh British? Britishh and British', 'What are the main focuses of the 2025 DoD budget spotlight?\nR focus? light? budget? the? main? focus?D focus? main? light? 2 5 2 2 4 4 5', 'What does the story of the Harlem Hellfighters celebrate?\ncere? celest?e celeb?rated? celebrating celebrat?ed? celebrator? celebration?\n#\n#\n#\n#']  
cr1='The Department of the Air Force has requested a budget of $217.5 billion for the fiscal year 2025. This request includes $188.1 billion allocated for the Air Force and $29.4 billion for the Space Force. The budget aims to continue modernizing both forces, maintain readiness to address current threats, and invest in addressing key capability gaps【15†source】【16†source】【17†source】.'
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="q1",
    # Replace this with the actual output from your LLM application
    actual_output="r1[0]",
    retrieval_context=cr1
)

answer_relevancy_metric.measure(test_case)
#hallucination_metric = HallucinationMetric(threshold=0.3)
#hallucination_metric.measure(test_case)
print(answer_relevancy_metric.score)
# Most metrics also offer an explanation
print(answer_relevancy_metric.reason)
#print(hallucination_metric.score)
#print(hallucination_metric.reason)

print('$$$$$$')
#FyY9tHKpnaW4RFHrb0N7x6rdfQ0d6CU1R5NgpKe4gdw=
test_case = LLMTestCase(
    input="q2",
    # Replace this with the actual output from your LLM application
    actual_output="r1[1]",
    retrieval_context=[a2]
)

answer_relevancy_metric.measure(test_case)
#hallucination_metric = HallucinationMetric(threshold=0.3)
#hallucination_metric.measure(test_case)
print(answer_relevancy_metric.score)
# Most metrics also offer an explanation
print(answer_relevancy_metric.reason)
# c1 = LLMTestCase(input=q1, actual_output=r1[0], context=[a1])
# c2 = LLMTestCase(input=q2, actual_output=r1[1], context=[a2])
# c3 = LLMTestCase(input=q3, actual_output=r1[2], context=[a3])
# c4 = LLMTestCase(input=q4, actual_output=r1[3], context=[a4])
# c5 = LLMTestCase(input=q5, actual_output=r1[4], context=[a5])
# dataset = EvaluationDataset(test_cases=[c1, c2,c3,c4,c5])

# @pytest.mark.parametrize(
#     "test_case",
#     dataset,
# )
# def test_customer_chatbot(test_case: LLMTestCase):
#     hallucination_metric = HallucinationMetric(threshold=0.3)
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
#     assert_test(test_case, [hallucination_metric, answer_relevancy_metric])






    
