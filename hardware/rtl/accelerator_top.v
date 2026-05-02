`timescale 1ns/1ps

module accelerator_top #(
    parameter N = 32
)(
    input clk,
    input reset,
    input start,

    input signed [7:0] activations [0:N-1],
    input signed [7:0] weights [0:N-1],

    output signed [31:0] outputs [0:N-1],
    output compute_done
);

wire load_weights;
wire stream_activations;

array_controller #(.N(N)) controller (

    .clk(clk),
    .reset(reset),
    .start(start),

    .load_weights(load_weights),
    .stream_activations(stream_activations),
    .compute_done(compute_done)

);

wire signed [31:0] systolic_out [0:N-1];

systolic_array #(.N(N)) array (

    .clk(clk),
    .reset(reset),

    .activations(activations),
    .weights(weights),

    .outputs(systolic_out)

);

wire signed [31:0] relu_out [0:N-1];

genvar i;

generate

for(i = 0; i < N; i = i + 1) begin : relu_stage

    relu relu_inst (

        .clk(clk),
        .data_in(systolic_out[i]),
        .data_out(relu_out[i])

    );

end

endgenerate

assign outputs = relu_out;

endmodule
