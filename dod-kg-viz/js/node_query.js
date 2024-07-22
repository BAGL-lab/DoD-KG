export const node_query_init = (div_id) => {

  const container = document.getElementById(div_id);
  
  // Add input container
  const input_container = node_query_create_inputbox();
  container.append(input_container);
}

const node_query_create_inputbox = () => {
  // input container
  const input_container = document.createElement("div");
  input_container.className = "node_query-input-container";

  // input box
  const input_group_container =document.createElement("div");
  const input_group_label = document.createElement("label");
  input_group_label.textContent = "Groups: ";
  const input_group_select = document.createElement("select");
  input_group_select.className = "node-query-input-select";

  // add select options
  input_group_select.ariaPlaceholder = "no thing";

  input_group_select.add(new Option("DoD", 0));
  input_group_select.add(new Option("Homeland Security", 1));
  input_group_select.add(new Option("SpaceForceDoD", 2));
  input_group_select.add(new Option("NASA", 3));
  input_group_select.add(new Option("NASA", 3));
  

  input_group_container.append(input_group_label);
  input_group_container.append(input_group_select);

  // keyword search
  const input_keyword_label = document.createElement("label");
  input_keyword_label.textContent = "Keyword: ";
  const input_keyword_text = document.createElement("input");
  input_keyword_text.type = "text";
  input_keyword_text.className = "node-query-input-text";

  input_container.append(input_group_container);
  input_container.append(input_keyword_label);
  input_container.append(input_keyword_text);

  return input_container;
}
