`timescale 1ns/1ps
module maxpool_parallel #(parameter N = 4) (
    input clk, input reset,
    input [(N*32)-1:0] row0_flat, input [(N*32)-1:0] row1_flat,
    output reg [((N/2)*32)-1:0] pooled_flat
);
    wire [31:0] pooled_comb [0:(N/2)-1];
    genvar i;
    generate
        for (i = 0; i < N/2; i = i + 1) begin : pool_units
            wire signed [31:0] a = row0_flat[(2*i)*32 +: 32];
            wire signed [31:0] b = row0_flat[(2*i+1)*32 +: 32];
            wire signed [31:0] c = row1_flat[(2*i)*32 +: 32];
            wire signed [31:0] d = row1_flat[(2*i+1)*32 +: 32];
            wire signed [31:0] max1 = (a > b) ? a : b;
            wire signed [31:0] max2 = (c > d) ? c : d;
            assign pooled_comb[i] = (max1 > max2) ? max1 : max2;
        end
    endgenerate
    integer k;
    always @(posedge clk) begin
        if (reset) pooled_flat <= 0;
        else for (k = 0; k < N/2; k = k + 1)
            pooled_flat[k*32 +: 32] <= pooled_comb[k];
    end
endmodule
