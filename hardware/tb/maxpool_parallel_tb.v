`timescale 1ns/1ps

module maxpool_parallel_tb;

reg clk;
reg reset;

// N=4: row buses are 4*32 = 128 bits
reg [127:0] row0;
reg [127:0] row1;

// pooled bus is (N/2)*32 = 64 bits
wire [63:0] pooled;

maxpool_parallel #(.N(4)) uut (
    .clk        (clk),
    .reset      (reset),
    .row0_flat  (row0),
    .row1_flat  (row1),
    .pooled_flat(pooled)
);

always #5 clk = ~clk;

initial begin
    clk   = 0;
    reset = 1;

    // Pack values into flat buses: index 0 = bits [31:0], index 1 = bits [63:32], etc.
    // row0 = [5, 3, 7, 1], row1 = [2, 8, 4, 6]
    row0[31:0]   = 32'sd5;
    row0[63:32]  = 32'sd3;
    row0[95:64]  = 32'sd7;
    row0[127:96] = 32'sd1;

    row1[31:0]   = 32'sd2;
    row1[63:32]  = 32'sd8;
    row1[95:64]  = 32'sd4;
    row1[127:96] = 32'sd6;

    #12 reset = 0;
    #10;

    // pool[0] = max(5,3,2,8) = 8
    // pool[1] = max(7,1,4,6) = 7
    $display("Pool0 = %0d (expect 8)", $signed(pooled[31:0]));
    $display("Pool1 = %0d (expect 7)", $signed(pooled[63:32]));

    $finish;
end

endmodule
