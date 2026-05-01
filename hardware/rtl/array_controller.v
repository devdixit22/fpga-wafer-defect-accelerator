module array_controller #(
    parameter N = 32
)(
    input clk,
    input reset,
    input start,

    output reg load_weights,
    output reg stream_activations,
    output reg compute_done
);

reg [15:0] cycle_count;

always @(posedge clk) begin

    if(reset) begin
        cycle_count <= 0;
        load_weights <= 0;
        stream_activations <= 0;
        compute_done <= 0;
    end

    else if(start) begin

        cycle_count <= cycle_count + 1;

        if(cycle_count < N)
            load_weights <= 1;
        else
            load_weights <= 0;

        if(cycle_count >= N && cycle_count < 2*N)
            stream_activations <= 1;
        else
            stream_activations <= 0;

        if(cycle_count == 3*N)
            compute_done <= 1;

    end

end

endmodule
