`timescale 1ns/1ps

module pe (
    input clk,
    input reset,

    input  signed [7:0]  activation_in,
    input  signed [7:0]  weight_in,
    input  signed [31:0] partial_sum_in,

    output reg signed [7:0]  activation_out,
    output reg signed [7:0]  weight_out,
    output reg signed [31:0] partial_sum_out
);

wire signed [15:0] mult;
assign mult = activation_in * weight_in;

always @(posedge clk) begin
    if (reset) begin
        activation_out  <= 0;
        weight_out      <= 0;
        partial_sum_out <= 0;
    end else begin
        activation_out  <= activation_in;
        weight_out      <= weight_in;
        // FIX: sign-extend 16-bit mult to 32-bit before accumulating
        partial_sum_out <= partial_sum_in + {{16{mult[15]}}, mult};
    end
end

endmodule
