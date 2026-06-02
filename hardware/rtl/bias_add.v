`timescale 1ns/1ps
module bias_add #(parameter N=4, parameter WIDTH=32) (
    input [(N*WIDTH)-1:0] data_in_flat,
    input [(N*WIDTH)-1:0] bias_flat,
    output [(N*WIDTH)-1:0] data_out_flat
);
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : b_rows
            assign data_out_flat[i*WIDTH +: WIDTH] = 
                $signed(data_in_flat[i*WIDTH +: WIDTH]) + 
                $signed(bias_flat[i*WIDTH +: WIDTH]);
        end
    endgenerate
endmodule
