module pe_chain #(
    parameter N = 4
)(
    input clk,
    input reset,

    input signed [7:0] activation_in,

    input signed [7:0] weight [0:N-1],

    input signed [31:0] partial_sum_in [0:N-1],

    output signed [7:0] activation_out,

    output signed [31:0] partial_sum_out [0:N-1]
);

wire signed [7:0] activation_wire [0:N];

assign activation_wire[0] = activation_in;
assign activation_out = activation_wire[N];

genvar i;

generate
    for(i = 0; i < N; i = i + 1) begin : pe_row

        pe pe_inst (

            .clk(clk),
            .reset(reset),

            .activation_in(activation_wire[i]),
            .weight(weight[i]),

            .partial_sum_in(partial_sum_in[i]),

            .activation_out(activation_wire[i+1]),
            .partial_sum_out(partial_sum_out[i])

        );

    end
endgenerate

endmodule
