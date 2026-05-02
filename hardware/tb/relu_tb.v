`timescale 1ns/1ps

module relu_tb;

reg clk;
reg signed [31:0] data_in;

wire signed [31:0] data_out;

relu uut (
    .clk(clk),
    .data_in(data_in),
    .data_out(data_out)
);

always #5 clk = ~clk;

initial begin

    clk = 0;

    $display("TIME\tINPUT\tOUTPUT");

    data_in = -10;
    #10 $display("%0t\t%d\t%d", $time, data_in, data_out);

    data_in = 5;
    #10 $display("%0t\t%d\t%d", $time, data_in, data_out);

    data_in = -3;
    #10 $display("%0t\t%d\t%d", $time, data_in, data_out);

    data_in = 12;
    #10 $display("%0t\t%d\t%d", $time, data_in, data_out);

    $finish;

end

endmodule
