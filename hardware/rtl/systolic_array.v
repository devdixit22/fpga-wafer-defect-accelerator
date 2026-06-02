`timescale 1ns/1ps

module systolic_array #(parameter N=4, parameter DATA_WIDTH=8, parameter ACC_WIDTH=32) (
    input  clk,
    input  reset,
    input  load_weights,
    input  [(N*N*DATA_WIDTH)-1:0] weights,
    input  [(N*DATA_WIDTH)-1:0]   activations,
    output [(N*ACC_WIDTH)-1:0]    outputs
);

    wire signed [DATA_WIDTH-1:0] a_wire [0:(N+1)*(N+1)-1];
    wire signed [ACC_WIDTH-1:0]  p_wire [0:(N+1)*(N+1)-1];

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : row
            assign a_wire[i*(N+1) + 0] = activations[((i+1)*DATA_WIDTH)-1 -: DATA_WIDTH];
            for (j = 0; j < N; j = j + 1) begin : col
                wire signed [DATA_WIDTH-1:0] w;
                
                if (i == 0) begin
                    assign p_wire[0*(N+1) + j] = 0;
                end
                
                assign w = weights[(((i*N)+j+1)*DATA_WIDTH)-1 -: DATA_WIDTH];

                pe pe_inst (
                    .clk            (clk),
                    .reset          (reset),
                    .activation_in  (a_wire[i*(N+1) + j]),
                    .weight_in      (w),
                    .partial_sum_in (p_wire[i*(N+1) + j]),
                    .activation_out (a_wire[i*(N+1) + j + 1]),
                    .partial_sum_out(p_wire[(i+1)*(N+1) + j])
                );
            end
            
            assign outputs[((i+1)*ACC_WIDTH)-1 -: ACC_WIDTH] = p_wire[N*(N+1) + i];
        end
    endgenerate

endmodule
