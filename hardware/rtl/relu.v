`timescale 1ns/1ps

module relu (
    input clk,
    input reset,
    input  signed [31:0] data_in,
    output reg signed [31:0] data_out
);

always @(posedge clk) begin
    if (reset)
        data_out <= 0;
    else
        data_out <= (data_in[31]) ? 32'sd0 : data_in;  // MSB = sign bit
end

endmodule
