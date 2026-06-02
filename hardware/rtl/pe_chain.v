`timescale 1ns/1ps

// FIX: Changed .weight() -> .weight_in() to match pe.v port name
module pe_chain #(
    parameter N = 4
)(
    input clk,
    input reset,

    input  signed [7:0] activation_in,
    input  [(N*8)-1:0] weight,
    input  [(N*32)-1:0] partial_sum_in,

    output signed [7:0]  activation_out,
    output [(N*32)-1:0] partial_sum_out
);

wire signed [7:0] activation_wire [0:N];

assign activation_wire[0] = activation_in;
assign activation_out     = activation_wire[N];

genvar i;
generate
    for (i = 0; i < N; i = i + 1) begin : pe_row
        pe pe_inst (
            .clk             (clk),
            .reset           (reset),
            .activation_in   (activation_wire[i]),
            .weight_in       (weight[((i+1)*8)-1 -: 8]),          // FIX: was .weight()
            .partial_sum_in  (partial_sum_in[((i+1)*32)-1 -: 32]),
            .activation_out  (activation_wire[i+1]),
            .weight_out      (),                   // unused in chain
            .partial_sum_out (partial_sum_out[((i+1)*32)-1 -: 32])
        );
    end
endgenerate

endmodule
