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

wire signed [15:0] mult;

assign mult = activation_in * weight_in;

always @(posedge clk) begin

    if(reset) begin
        activation_out <= 0;
        weight_out <= 0;
        partial_sum_out <= 0;
    end

    else begin

        // propagate data through systolic array
        activation_out <= activation_in;
        weight_out <= weight_in;

        // accumulate convolution result
        partial_sum_out <= partial_sum_in + mult;

    end

end

endmodule
