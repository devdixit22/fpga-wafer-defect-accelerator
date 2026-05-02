`timescale 1ns/1ps

module activation_buffer #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 1024,
    parameter ADDR_WIDTH = 10
)(
    input clk,

    input write_en,
    input [ADDR_WIDTH-1:0] write_addr,
    input signed [DATA_WIDTH-1:0] write_data,

    input read_en,
    input [ADDR_WIDTH-1:0] read_addr,
    output reg signed [DATA_WIDTH-1:0] read_data
);

reg signed [DATA_WIDTH-1:0] memory [0:DEPTH-1];


// ADD THIS BLOCK HERE
initial begin
    $readmemh("activation.mem", memory);
end


always @(posedge clk) begin

    if(write_en)
        memory[write_addr] <= write_data;

    if(read_en)
        read_data <= memory[read_addr];

end

endmodule
