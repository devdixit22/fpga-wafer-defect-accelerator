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

assign outputs = sys_out;

endmodule
