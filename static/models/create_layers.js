// var layers = {
//     "Conv2d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
//     "Linear": ["in_features", "out_features", "bias"],
//     "Loss": ["loss_function"]
// }
// var parameters = {
//     "filter": filter,
//     "kernel_size": kernel_size,
//     "padding": padding,
//     "strides": strides,
//     "activate": activate,
//     "loss_function": loss_function
// }
//
//
// $(document).ready(function () {
//     $("#layer_selector").hide();
//     $("#layer_closer button").click(function () {
//         $("#layer_selector").toggle();
//     })
//     $("#add_layer").click(function () {
//         $("#layer_selector").toggle();
//     })
//
//     $(".layer").click(function () {
//         $("#add_layer").before(create_layer(this.value));
//         $("#layer_selector").toggle();
//     })
// })
//
// function create_layer(name) {
//     let html = $("<form></form>").attr("name", name)
//
//     let para = $("<div></div>").hide()
//     for (let i = 0; i < layers[name].length; i++) {
//         para.append(parameters[layers[name][i]]());
//     }
//
//     let button = $("<input>").attr("type", "button")
//         .attr("name", name)
//         .attr("value", name)
//         .addClass("layer")
//         .addClass("btn btn-primary btn-lg btn-block mb-1")
//         .click(function () {
//             para.toggle()
//         })
//
//     html.append(button);
//     html.append(para);
//     return html;
// }
//
//
// function activate() {
//     let activate = $("<div></div>").addClass("ml-5");
//     let label = $("<label>activate</label>").addClass("mr-2")
//     activate.append(label)
//     let select = $("<select></select>").attr("name", "activate");
//     select.append($("<option>ReLU</option>"));
//     select.append($("<option>Softmax</option>"));
//     select.append($("<option>Tanh</option>"));
//     activate.append(select)
//     return activate
// }
//
// function kernel_size() {
//     let html = $("<div></div>").addClass("ml-5");
//     let label = $("<label>kernel_size</label>").addClass("mr-2");
//     html.append(label)
//     let kernel = $("<input>").attr("type", "text")
//         .attr("name", "kernel_size");
//     html.append(kernel)
//     return html
// }
//
// function filter() {
//     let html = $("<div></div>").addClass("ml-5");
//     let label = $("<label>filter</label>").addClass("mr-2");
//     html.append(label);
//     let filter = $("<input>").attr("type", "text")
//         .attr("name", "filter");
//     html.append(filter);
//     return html
// }
//
// function padding() {
//     let html = $("<div></div>").addClass("ml-5");
//     let label = $("<label>padding</label>").addClass("mr-2")
//     html.append(label)
//     let select = $("<select></select>").attr("name", "padding");
//     select.append($("<option>valid</option>"));
//     select.append($("<option>same</option>"));
//     html.append(select)
//     return html
// }
//
// function strides() {
//     let html = $("<div></div>").addClass("ml-5");
//     let label = $("<label>strides</label>").addClass("mr-2");
//     html.append(label)
//     let kernel = $("<input>").attr("type", "text")
//         .attr("name", "strides");
//     html.append(kernel)
//     return html
// }
//
// function loss_function() {
//     let html = $("<div></div>").addClass("ml-5");
//     let label = $("<label>loss_function</label>").addClass("mr-2")
//     html.append(label)
//     let select = $("<select></select>").attr("name", "loss_function");
//     select.append($("<option>CategoricalCrossentropy</option>"));
//     select.append($("<option>BinaryCrossentropy</option>"));
//     html.append(select)
//     return html
// }

const data = {
    // ??????
    nodes: [
        {
            id: 'node1', // String?????????????????????????????????
            x: 40,       // Number??????????????????????????? x ???
            y: 40,       // Number??????????????????????????? y ???
            width: 80,   // Number??????????????????????????? width ???
            height: 40,  // Number??????????????????????????? height ???
            label: 'hello', // String???????????????
        },
        {
            id: 'node2', // String????????????????????????
            x: 160,      // Number??????????????????????????? x ???
            y: 180,      // Number??????????????????????????? y ???
            width: 80,   // Number??????????????????????????? width ???
            height: 40,  // Number??????????????????????????? height ???
            label: 'world', // String???????????????
        },
    ],
    // ???
    edges: [
        {
            source: 'node1', // String???????????????????????? id
            target: 'node2', // String???????????????????????? id
        },
    ],
};
const graph = new X6.Graph({
    container: $(".model-build")[0],
    grid: {
        size: 10,
        visible: true
    },
    snapline: true,
    scroller: true,
    // minimap: true,
    history: true,
    selecting: true,
    connecting: {
        anchor: {
            name: "midSide",
            rotate: false
        },
        allowBlank: false,
        allowLoop: false,
        highlight: true,
        connector: "normal",
        router: {
            name: "manhattan",
            args: {
                step: 10,
                offset: "center",
                endDirections: ["top", "left", "right"],
                startDirections: ["bottom"],
                excludeTerminals: ["source", "target"],
            }
        }
    },
    connector: {
        name: "smooth"
    }
})

function create_node(x, y) {
    const rect = new X6.Shape.Rect({
        x: x,
        y: y,
        width: 100,
        height: 40,
        label: 'rect',
        zIndex: 2,
        ports: {
            groups: {
                in: {
                    position: "top",
                    attrs: {
                        circle: {
                            r: 5,
                            magnet: true,
                            stroke: "#31d0c6",
                            strokeWidth: 2,
                            fill: "#fff"
                        }
                    }
                },
                out: {
                    position: "bottom",
                    attrs: {
                        circle: {
                            r: 5,
                            magnet: true,
                            stroke: "#31d0c6",
                            strokeWidth: 2,
                            fill: "#fff"
                        }
                    }
                }
            },
            items: [
                {
                    group: "in",

                },
                {
                    group: "out",

                }
            ]
        }
    })
    return rect
}

graph.fromJSON(data)

graph.on('blank:click', ({e, x, y}) => {
    graph.addNode(create_node(x, y))
    console.log(graph.toJSON())
})



