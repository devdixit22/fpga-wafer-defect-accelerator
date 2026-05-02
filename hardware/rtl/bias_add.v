`timescale 1ns/1ps

module bias_add #(
    parameter N = 32
)(
    input clk,
    input reset,

    input signed [31:0] data_in [0:N-1],
    input signed [31:0] bias [0:N-1],

    output reg signed [31:0] data_out [0:N-1]
);

integer i;

always @(posedge clk) begin
    if(reset) begin
        for(i=0;i<N;i=i+1)
            data_out[i] <= 0;
    end
    else begin
        for(i=0;i<N;i=i+1)
            data_out[i] <= data_in[i] + bias[i];
    end
end

endmodule
