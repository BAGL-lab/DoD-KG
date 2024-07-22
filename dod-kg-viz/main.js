import "./css/style.css";
import "./css/node_query.css";

import * as d3 from "d3";
import { draw_nodelink } from "./js/node_link_expander";
import { preprocess } from "./js/preprocess";
import crossfilter from "crossfilter2";
import { node_query_init } from "./js/node_query";
import { draw_node_link_expander } from "./js/relationship_expander";

const KG_DATA_PATH = "../data/KG.json";
let KG_DATA = undefined;
let KG_crossfilter = undefined;

const initialize_crossfilter = (data) => {
  
  const _crossfilter = crossfilter([]);
  // add nodes and links
  _crossfilter.add(data.nodes.map(d => ({ type: 'node', ...d })));
  _crossfilter.add(data.links.map(d => ({ type: 'link', ...d })));
  
  return _crossfilter;
}

export const get_KG_DATA = () => {
  return (KG_DATA) ? KG_DATA : undefined;
}

const initialize = () => {
  d3.json(KG_DATA_PATH).then(function(data) {
    KG_DATA = preprocess(data);
    KG_crossfilter = initialize_crossfilter(KG_DATA);
    node_query_init("node-query-body");
    
    // console.log(KG_DATA);
    // console.log(KG_crossfilter);
    
    draw_nodelink("node-expander-body", KG_DATA);
    draw_node_link_expander("node-relationship-expander", KG_DATA);
  });
}

window.onload = initialize();

export {
  KG_DATA,
}
