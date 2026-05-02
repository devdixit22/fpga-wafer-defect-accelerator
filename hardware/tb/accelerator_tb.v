module accelerator_tb;

reg clk;
reg reset;

wire signed [31:0] result;

accelerator_top DUT (
    .clk(clk),
    .reset(reset),
    .result(result)
);

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    reset = 1;
    #20 reset = 0;
end

integer fd;

initial begin

    fd = $fopen("output.mem","w");

    #2000;

    $fdisplay(fd,"%d", result);

    $fclose(fd);

    $finish;

end

endmodule
