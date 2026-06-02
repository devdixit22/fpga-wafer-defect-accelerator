`timescale 1ns/1ps

module system_top #(
    parameter N = 4,
    parameter IN_BLOCKS = 3,    // padded in_features / 4
    parameter OUT_BLOCKS = 1,   // out_channels / 4
    parameter NUM_PATCHES = 256 // spatial patches
)(
    input clk,
    input reset,
    input start,
    output reg done
);

    // Memory connections
    wire [11:0] act_addr;
    wire [(N*8)-1:0] act_data;

    wire [11:0] wgt_addr;
    wire [(N*N*8)-1:0] wgt_data;

    wire [15:0] out_addr;
    reg  [(N*32)-1:0] out_data;
    reg  out_we;

    // Buffers
    reg [15:0] out_addr_reg;

    activation_buffer #(
        .DATA_WIDTH(N*8),
        .DEPTH(4096),
        .ADDR_WIDTH(12)
    ) act_buf (
        .clk(clk),
        .write_en(1'b0), .write_addr(12'd0), .write_data(0),
        .read_en(1'b1), .read_addr(act_addr), .read_data(act_data)
    );

    weight_buffer #(
        .DATA_WIDTH(N*N*8),
        .DEPTH(4096),
        .ADDR_WIDTH(12)
    ) wgt_buf (
        .clk(clk),
        .write_en(1'b0), .write_addr(12'd0), .write_data(0),
        .read_en(1'b1), .read_addr(wgt_addr), .read_data(wgt_data)
    );

    output_buffer #(
        .DATA_WIDTH(N*32),
        .DEPTH(16384),
        .ADDR_WIDTH(16)
    ) out_buf (
        .clk(clk),
        .write_en(out_we), .write_addr(out_addr_reg), .write_data(out_data),
        .read_en(1'b0), .read_addr(16'd0), .read_data()
    );

    // Accelerator connection
    reg accel_start;
    wire accel_done;
    wire [(N*32)-1:0] accel_out;

    accelerator_top #(.N(N)) accel (
        .clk(clk),
        .reset(reset),
        .start(accel_start),
        .weights(wgt_data),
        .activations(act_data),
        .bias({(N*32){1'b0}}), // Bias handled in software or zeroed out here
        .outputs(accel_out),
        .compute_done(accel_done)
    );

    // State machine
    reg [2:0] state;
    localparam IDLE = 0, FETCH = 1, WAIT_ACCEL = 2, ACCUM = 3, NEXT = 4;

    reg [15:0] out_b;
    reg [15:0] patch;
    reg [15:0] in_b;

    reg signed [31:0] acc [0:N-1];

    assign wgt_addr = out_b * IN_BLOCKS + in_b;
    assign act_addr = patch * IN_BLOCKS + in_b;
    assign out_addr = out_b * NUM_PATCHES + patch;

    integer i;

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done <= 0;
            out_b <= 0;
            patch <= 0;
            in_b <= 0;
            accel_start <= 0;
            out_we <= 0;
            for (i=0; i<N; i=i+1) acc[i] <= 0;
        end else begin
            accel_start <= 0;
            out_we <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        state <= FETCH;
                        out_b <= 0;
                        patch <= 0;
                        in_b <= 0;
                        for (i=0; i<N; i=i+1) acc[i] <= 0;
                    end
                end

                FETCH: begin
                    // memory read takes 1 cycle, so accel_start is next cycle
                    // Wait, BRAM read is 1 cycle. In cycle FETCH we output address.
                    // Data is ready next cycle.
                    state <= WAIT_ACCEL;
                    accel_start <= 1; // Assert start next cycle when data is valid?
                    // Actually, if we assert accel_start now, accelerator_top goes to LOADING state,
                    // which takes N cycles. So data just needs to be valid.
                end

                WAIT_ACCEL: begin
                    if (accel_done) begin
                        state <= ACCUM;
                    end
                end

                ACCUM: begin
                    for (i=0; i<N; i=i+1) begin
                        acc[i] <= acc[i] + $signed(accel_out[((i+1)*32)-1 -: 32]);
                    end
                    
                    if (in_b == IN_BLOCKS - 1) begin
                        state <= NEXT;
                        for (i=0; i<N; i=i+1) begin
                            out_data[((i+1)*32)-1 -: 32] <= acc[i] + $signed(accel_out[((i+1)*32)-1 -: 32]);
                        end
                        out_addr_reg <= out_addr;
                    end else begin
                        in_b <= in_b + 1;
                        state <= FETCH;
                    end
                end

                NEXT: begin
                    out_we <= 1;

                    if (patch == NUM_PATCHES - 1) begin
                        patch <= 0;
                        if (out_b == OUT_BLOCKS - 1) begin
                            done <= 1;
                            state <= IDLE;
                        end else begin
                            out_b <= out_b + 1;
                            state <= FETCH;
                            for (i=0; i<N; i=i+1) acc[i] <= 0;
                        end
                    end else begin
                        patch <= patch + 1;
                        state <= FETCH;
                        in_b <= 0;
                        for (i=0; i<N; i=i+1) acc[i] <= 0;
                    end
                end
            endcase
        end
    end
endmodule
