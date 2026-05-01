module pe (
    input clk,
    input reset,

    // activation input from left PE
    input signed [7:0] activation_in,

    // weight stored inside PE
    input signed [7:0] weight,

    // partial sum from upper PE
    input signed [31:0] partial_sum_in,

    // activation forwarded to next PE
    output reg signed [7:0] activation_out,

    // partial sum forwarded downward
    output reg signed [31:0] partial_sum_out
);

reg signed [15:0] mult_result;

always @(posedge clk) begin
    if (reset) begin
        activation_out <= 0;
        partial_sum_out <= 0;
    end
    else begin
        mult_result = activation_in * weight;

        partial_sum_out <= partial_sum_in + mult_result;

        activation_out <= activation_in;
    end
end

endmodule
