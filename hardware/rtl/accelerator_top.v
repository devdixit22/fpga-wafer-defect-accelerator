`timescale 1ns/1ps

// FIX: All array ports replaced with flat packed buses throughout.
module accelerator_top #(
    parameter N = 4
)(
    input  clk,
    input  reset,
    input  start,

    input  [(N*N*8)-1:0]  weights,      // N x N weight matrix packed
    input  [(N*8)-1:0]    activations,  // N activations packed
    input  [(N*32)-1:0]   bias,         // N biases packed

    output [(N*32)-1:0]   outputs,
    output compute_done
);

// Controller
wire load_weights_en;
wire stream_activations;

array_controller #(.N(N)) ctrl (
    .clk               (clk),
    .reset             (reset),
    .start             (start),
    .load_weights      (load_weights_en),
    .stream_activations(stream_activations),
    .compute_done      (compute_done)
);

// Systolic array
wire [(N*32)-1:0] sys_out;

systolic_array #(.N(N)) array (
    .clk          (clk),
    .reset        (reset),
    .load_weights (load_weights_en),
    .weights      (weights),
    .activations  (activations),
    .outputs      (sys_out)
);

// Bias add
wire [(N*32)-1:0] biased;

bias_add #(.N(N)) bias_unit (
    .data_in_flat  (sys_out),
    .bias_flat     (bias),
    .data_out_flat (biased)
);

// ReLU stage
wire [(N*32)-1:0] relu_out;

genvar i;
generate
    for (i = 0; i < N; i = i + 1) begin : relu_stage
        relu relu_inst (
            .clk      (clk),
            .reset    (reset),
            .data_in  (biased[((i+1)*32)-1 : i*32]),
            .data_out (relu_out[((i+1)*32)-1 : i*32])
        );
    end
endgenerate

assign outputs = relu_out;

endmodule
