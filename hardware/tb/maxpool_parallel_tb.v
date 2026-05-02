`timescale 1ns/1ps

module maxpool_parallel_tb;

reg clk;

reg signed [31:0] row0 [0:3];
reg signed [31:0] row1 [0:3];

wire signed [31:0] pooled [0:1];

maxpool_parallel #(.N(4)) uut (

    .clk(clk),

    .row0(row0),
    .row1(row1),

    .pooled(pooled)

);

always #5 clk = ~clk;

initial begin

    clk = 0;

    row0[0] = 5;
    row0[1] = 3;
    row0[2] = 7;
    row0[3] = 1;

    row1[0] = 2;
    row1[1] = 8;
    row1[2] = 4;
    row1[3] = 6;

    #10;

    $display("Pool0 = %d", pooled[0]);
    $display("Pool1 = %d", pooled[1]);

    $finish;

end

endmodule
