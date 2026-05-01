`timescale 1ns/1ps

module pe_tb;

reg clk;
reg reset;

reg signed [7:0] activation_in;
reg signed [7:0] weight;
reg signed [31:0] partial_sum_in;

wire signed [7:0] activation_out;
wire signed [31:0] partial_sum_out;

pe uut (
    .clk(clk),
    .reset(reset),
    .activation_in(activation_in),
    .weight(weight),
    .partial_sum_in(partial_sum_in),
    .activation_out(activation_out),
    .partial_sum_out(partial_sum_out)
);

always #5 clk = ~clk;

initial begin
    clk = 0;
    reset = 1;

    activation_in = 0;
    weight = 3;
    partial_sum_in = 10;

    #10 reset = 0;

    activation_in = 5;

    #20;

    $display("Result: %d", partial_sum_out);

    #20 $finish;
end

endmodule
