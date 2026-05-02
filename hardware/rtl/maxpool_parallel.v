`timescale 1ns/1ps

module maxpool_parallel #(
    parameter N = 32
)(
    input clk,

    input signed [31:0] row0 [0:N-1],
    input signed [31:0] row1 [0:N-1],

    output signed [31:0] pooled [0:(N/2)-1]
);

genvar i;

generate

for(i = 0; i < N/2; i = i + 1) begin : pool_units

    wire signed [31:0] a = row0[2*i];
    wire signed [31:0] b = row0[2*i+1];
    wire signed [31:0] c = row1[2*i];
    wire signed [31:0] d = row1[2*i+1];

    wire signed [31:0] max1;
    wire signed [31:0] max2;

    assign max1 = (a > b) ? a : b;
    assign max2 = (c > d) ? c : d;

    assign pooled[i] = (max1 > max2) ? max1 : max2;

end

endgenerate

endmodule
