`timescale 1ns/1ps

module pe_tb;

reg clk;
reg reset;

reg signed [7:0] activation;
reg signed [7:0] weight;
reg signed [31:0] partial_sum_in;

wire signed [31:0] partial_sum_out;

pe uut (
    .clk(clk),
    .reset(reset),
    .activation(activation),
    .weight(weight),
    .partial_sum_in(partial_sum_in),
    .partial_sum_out(partial_sum_out)
);

always #5 clk = ~clk;

initial begin

    clk = 0;
    reset = 1;

    activation = 0;
    weight = 0;
    partial_sum_in = 0;

    #10 reset = 0;

    activation = 5;
    weight = 3;
    partial_sum_in = 10;

    #20;

    $display("Result: %d", partial_sum_out);

    #20 $finish;
end

endmodule
