import * as d3 from "d3";

var curve = function(context) {
  var custom = d3.curveLinear(context);
  custom._context = context;
  custom.point = function(x,y) {
    x = +x, y = +y;
    switch (this._point) {
      case 0: this._point = 1; 
        this._line ? this._context.lineTo(x, y) : this._context.moveTo(x, y);
        this.x0 = x; this.y0 = y;
        break;
      case 1: this._point = 2;
      default: 
        if (Math.abs(this.x0 - x) > Math.abs(this.y0 - y)) {
           var x1 = this.x0 * 0.5 + x * 0.5;
           this._context.bezierCurveTo(x1,this.y0,x1,y,x,y); 
        }
        else {
           var y1 = this.y0 * 0.5 + y * 0.5;
           this._context.bezierCurveTo(this.x0,y1,x,y1,x,y);
        }
        this.x0 = x; this.y0 = y; 
        break;
    }
  }
  return custom;
}

export function draw_nodelink(div_id, data) {

  const margin = {top: 0, right: 0, bottom: 0  , left: 0};
  const element = d3.select("#" + div_id).node();
  const width = element.getBoundingClientRect().width - margin.left - margin.right;
  const height = element.getBoundingClientRect().height - margin.top - margin.bottom;

  const zoom = d3.zoom()
    .scaleExtent([0.5, 10])
    .on("zoom", zoomed);

  const svg = d3.select("#" + div_id)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  const line = d3.line().curve(curve);
  const nodes = data["nodes"];
  const links = data["links"];

  var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(function(d) { return d.id; }))
    .force("charge", d3.forceManyBody().strength(-8))
    //.force("collide", d3.forceCollide(12))
    .force("center", d3.forceCenter(300, 200));

  const color_scale = d3.scaleOrdinal()
    .domain([0, 8])
    .range(['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc','#cccccc']);

  var link = svg.append("g")
    .selectAll("path")
    .data(links)
    .enter().append("path");

  const node = svg.append("g")
    .selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("r", 8)
    .attr("fill", d => color_scale(d.group))
    .attr("stroke", "#525252")
    .style("cursor", "pointer")
    .on("mouseover", function(evt, d) {
      // set text as tooltip
      tooltip.text(d.keyword);
      return tooltip.style("visibility", "visible");
    })
    .on("mousemove", function() {
      return tooltip.style("top", (event.pageY)+"px").style("left",(event.pageX + 20)+"px");
    })
    .on("mouseout", function() {
      return tooltip.style("visibility", "hidden");
    });

  svg.call(d3.drag()
  .on("start", dragstarted)
  .on("drag", dragged)
  .on("end", dragended));

  svg.call(d3.zoom()
    .extent([[0, 0], [width, height]])
    .scaleExtent([1, 8])
    .on("zoom", zoomed));

  simulation
    .nodes(nodes)
    .on("tick", ticked)
    .force("link")
    .links(links);

  // create a tooltip
  var tooltip = d3.select("#" + div_id)
    .append("div")
    .style("position", "absolute")
    .style("visibility", "hidden")
    .style("padding", "2px")
    .style("background-color", "#f0f0f0")
    .style("z-index", 1000)
    .text("I'm tooltip");

  function ticked() {
    link.attr("d", function(d) {
      return line([[d.source.x,d.source.y],[d.target.x,d.target.y]]);
    })

    node
      .attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; });
  }

  function dragstarted(event) {
    d3.select(this).raise().classed("active", true);
  }

  function dragged(event, d) {
    d3.select(this).attr("transform", "translate(" + event.x + "," + event.y + ")");
  }

  function dragended(event) {
    d3.select(this).classed("active", false);
  }

  function zoomed({transform}) {
    svg.attr("transform", transform);
  }
}
