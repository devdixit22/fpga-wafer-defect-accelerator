module systolic_array #(
    parameter N = 32
)(
    input clk,
    input reset,

    input signed [7:0] activations [0:N-1],
    input signed [7:0] weights [0:N-1],

    output signed [31:0] outputs [0:N-1]
);

wire signed [7:0] activation_wire [0:N-1][0:N];
wire signed [7:0] weight_wire [0:N][0:N-1];
wire signed [31:0] psum_wire [0:N][0:N-1];

genvar i, j;

generate

for(i = 0; i < N; i = i + 1) begin

    assign activation_wire[i][0] = activations[i];

end

for(j = 0; j < N; j = j + 1) begin

    assign weight_wire[0][j] = weights[j];

end


for(i = 0; i < N; i = i + 1) begin
for(j = 0; j < N; j = j + 1) begin : pe_grid

    pe pe_inst (

        .clk(clk),
        .reset(reset),

        .activation_in(activation_wire[i][j]),
        .weight_in(weight_wire[i][j]),

        .partial_sum_in(psum_wire[i][j]),

        .activation_out(activation_wire[i][j+1]),
        .weight_out(weight_wire[i+1][j]),

        .partial_sum_out(psum_wire[i+1][j])

    );

end
end

endgenerate

generate

for(i = 0; i < N; i = i + 1) begin

    assign outputs[i] = psum_wire[N][i];

end

endgenerate

endmodule
