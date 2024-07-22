import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal } from 'd3-sankey'

export function draw_node_link_expander (div_id, sankey_data) {
  
  var element = d3.select("#" + div_id).node();
  var margin = {top: 10, right: 10, bottom: 10, left: 10};
  var width = element.getBoundingClientRect().width - margin.left - margin.right;
  var height = element.getBoundingClientRect().height - margin.top - margin.bottom;

  const color = d3.scaleOrdinal()
    .domain([0, 8])
    .range(['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc','#cccccc']);

  var svg = d3.select("#" + div_id).append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;")
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  const nodeWidth = 50;
  const nodePadding = 40;

  const { nodes, links } = sankey()
    .nodeWidth(nodeWidth)
    .nodePadding(nodePadding)
    .extent([[1, 1], [width - 1, height - 25]])({
        nodes: sankey_data.nodes,
        links: sankey_data.links
    });

  const color_scale = d3.scaleOrdinal()
    .domain([0, 8])
    .range(['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc','#cccccc']);

  const link = svg.append('g')
    .attr('fill', 'none')
    .attr('stroke-opacity', 0.5)
    .selectAll('g')
    .data(links)
    .enter()
    .append('g')
    .style('mix-blend-mode', 'multiply')

  const gradient = link.append('linearGradient')
    .attr('gradientUnits', 'userSpaceOnUse')
    .attr('x1', d => d.source.x1)
    .attr('x2', d => d.target.x0);

  gradient.append('stop')
    .attr('offset', '0%')
    .attr('stop-color', d => color(d.source.group))

  gradient.append('stop')
    .attr('offset', '100%')
    .attr('stop-color', d => color(d.target.group))

  link.append('path')
    .attr('d', sankeyLinkHorizontal())
    .attr('stroke', "#000")
    .attr('stroke-width', d => Math.max(1, d.width))

  svg.append('g')
    .attr('stroke', '#000')
    .selectAll('rect')
    .data(nodes)
    .enter()
    .append('circle')
    .attr('cx', d => (d.x0 < width / 2 ? d.x1 - 5 : d.x0 + 5))
    .attr('cy', d => (d.y0 + d.y1) / 2)
    .attr('r', 5)
    .attr('fill', d => color_scale(d.group))
    .style('cursor', 'pointer');

  return;
}
