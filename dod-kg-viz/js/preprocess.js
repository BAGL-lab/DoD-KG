export const preprocess = (data) => {
  const nodes = data["nodes"];
  const links = data["links"];
  const preprocessed_nodes = preprocess_nodes(nodes);
  const preprocessed_links = preprocess_links(links, preprocessed_nodes);
  return {
    nodes: preprocessed_nodes,
    links: preprocessed_links,
  };
};

const preprocess_nodes = (nodes) => {
  const preprocessed_nodes = [];
  for (let i = 0; i < nodes.length; ++i) {
    const node = nodes[i];
    const keyword = node["Keywords"][0][0];
    const score = node["Keywords"][0][1];
    if (score >= 0.5) {
      const node_dict = {};
      node_dict["id"] = node["id"];
      node_dict["group"] = node["Group"];
      node_dict["group_name"] = node["Group Name"];
      node_dict["keyword"] = keyword;
      node_dict["score"] = score;
      node_dict["tweet"] = node["Original Tweet"];
      preprocessed_nodes.push(node_dict);
    }
  }
  return preprocessed_nodes;
};

const preprocess_links = (links, nodes) => {
  const preprocessed_links = [];
  for (let i = 0; i < links.length; ++i) {
    const link = links[i];
    const source = link["source"];
    const target = link["target"];
    const value = link["Similarity"];
    // filter link similarity >= 0.5
    // we also need to filter existing nodes (filtered)
    const target_index = nodes.map(x => x.id).indexOf(target);
    const source_index = nodes.map(x => x.id).indexOf(source);
    if (value >= 0.5 && target_index >= 0 && source_index >= 0) {
      const link_dict = {};
      link_dict["target"] = link["target"];
      link_dict["source"] = link["source"];
      link_dict["value"] = value;
      preprocessed_links.push(link_dict);
    }
  }
  return preprocessed_links;
};
