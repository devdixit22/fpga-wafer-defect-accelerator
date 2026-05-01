module pe (

    input clk,
    input reset,

    input signed [7:0] activation_in,
    input signed [7:0] weight_in,

    input signed [31:0] partial_sum_in,

    output reg signed [7:0] activation_out,
    output reg signed [7:0] weight_out,

    output reg signed [31:0] partial_sum_out
);

reg signed [15:0] mult_result;

always @(posedge clk) begin
    if (reset) begin

        activation_out <= 0;
        weight_out <= 0;
        partial_sum_out <= 0;

    end
    else begin

        mult_result = activation_in * weight_in;

        partial_sum_out <= partial_sum_in + mult_result;

        activation_out <= activation_in;

        weight_out <= weight_in;

    end
end

endmodule
