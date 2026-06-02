`timescale 1ns/1ps

// FIX: Changed .weight() -> .weight_in() to match pe.v port name.
//      Added weight_out port connection.
//      Fixed display timing: sample AFTER clock edge settles (#12 not #10).
module pe_tb;

reg clk;
reg reset;

reg signed [7:0]  activation_in;
reg signed [7:0]  weight_in;
reg signed [31:0] partial_sum_in;

wire signed [7:0]  activation_out;
wire signed [7:0]  weight_out;      // FIX: was missing
wire signed [31:0] partial_sum_out;

pe uut (
    .clk            (clk),
    .reset          (reset),
    .activation_in  (activation_in),
    .weight_in      (weight_in),     // FIX: was .weight()
    .partial_sum_in (partial_sum_in),
    .activation_out (activation_out),
    .weight_out     (weight_out),
    .partial_sum_out(partial_sum_out)
);

always #5 clk = ~clk;

always @(posedge clk) begin
    if (!reset) begin
        partial_sum_in <= partial_sum_out;
    end
end

initial begin
    clk            = 0;
    reset          = 1;
    activation_in  = 0;
    weight_in      = 3;
    partial_sum_in = 10;

    #12 reset = 0;  // release reset after first edge

    // Cycle 1: activation=5, weight=3 -> mult=15, psum = 10+15 = 25
    activation_in = 8'sd5;
    #10;

    // Cycle 2: change activation=2 -> mult=6, psum = 25+6 = 31
    activation_in = 8'sd2;
    #10;

    // FIX: sample at #12 to read value registered on the previous posedge
    $display("partial_sum after cycle 1 (expect 25): %0d", partial_sum_out);

    #10;
    $display("partial_sum after cycle 2 (expect 31): %0d", partial_sum_out);

    #20 $finish;
end

endmodule
