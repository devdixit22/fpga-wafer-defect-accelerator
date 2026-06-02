`timescale 1ns/1ps

// FIX: Added reset port connection.
//      Fixed sampling timing - display at #12 (after posedge at #10) not #10.
module relu_tb;

reg clk;
reg reset;
reg signed [31:0] data_in;
wire signed [31:0] data_out;

relu uut (
    .clk     (clk),
    .reset   (reset),
    .data_in (data_in),
    .data_out(data_out)
);

always #5 clk = ~clk;

initial begin
    clk    = 0;
    reset  = 1;
    data_in = 0;

    #12 reset = 0;

    $display("TIME\t\tINPUT\tOUTPUT");

    data_in = -10;
    #10; $display("%0t\t%0d\t%0d (expect 0)",  $time, data_in, data_out);

    data_in = 5;
    #10; $display("%0t\t%0d\t%0d (expect 5)",  $time, data_in, data_out);

    data_in = -3;
    #10; $display("%0t\t%0d\t%0d (expect 0)",  $time, data_in, data_out);

    data_in = 12;
    #10; $display("%0t\t%0d\t%0d (expect 12)", $time, data_in, data_out);

    $finish;
end

endmodule
