module pe (
    input clk,
    input reset,

    input signed [7:0] activation,
    input signed [7:0] weight,

    input signed [31:0] partial_sum_in,
    output reg signed [31:0] partial_sum_out
);

reg signed [15:0] mult_result;

always @(posedge clk) begin
    if (reset) begin
        partial_sum_out <= 0;
    end
    else begin
        mult_result = activation * weight;
        partial_sum_out <= partial_sum_in + mult_result;
    end
end

endmodule
