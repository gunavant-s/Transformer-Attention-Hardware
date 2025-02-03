//---------------------------------------------------------------------------
// Filename    : dut.sv
// Author      : Gunavant Setty
// Affiliation : North Carolina State University, Raleigh, NC
// Date        : Nov 2024
// Email       : gsetty@ncsu.edu, gunavantsetty26@gmail.com
// Description : The design successfully realizes the complete self-attention computation,
//               including matrix multiplications for Q, K, and V matrices, score(S), and 
//               attention(Z) calculation. Effective utilization of scratchpad for caculating
//               for caculating attention and score matrices.
//---------------------------------------------------------------------------
// DUT - 564 Project
//---------------------------------------------------------------------------
`include "common.vh"

module MyDesign(
	//---------------------------------------------------------------------------
	//System signals
	input wire reset_n                      ,  
	input wire clk                          ,

	//---------------------------------------------------------------------------
	//Control signals
	input wire dut_valid                    , 
	output wire dut_ready                   ,

	//---------------------------------------------------------------------------
	//input SRAM interface
	output wire                                  dut__tb__sram_input_write_enable  ,
	output wire        [`SRAM_ADDR_RANGE     ]   dut__tb__sram_input_write_address ,
	output wire signed [`SRAM_DATA_RANGE     ]   dut__tb__sram_input_write_data    ,
	output wire        [`SRAM_ADDR_RANGE     ]   dut__tb__sram_input_read_address  , 
	input  wire signed [`SRAM_DATA_RANGE     ]   tb__dut__sram_input_read_data     ,     

	//weight SRAM interface
	output wire                                  dut__tb__sram_weight_write_enable  ,
	output wire        [`SRAM_ADDR_RANGE     ]   dut__tb__sram_weight_write_address ,
	output wire signed [`SRAM_DATA_RANGE     ]   dut__tb__sram_weight_write_data    ,
	output wire        [`SRAM_ADDR_RANGE     ]   dut__tb__sram_weight_read_address  , 
	input  wire signed [`SRAM_DATA_RANGE     ]   tb__dut__sram_weight_read_data     ,     

	//result SRAM interface
	output wire                                  dut__tb__sram_result_write_enable  ,
	output wire        [`SRAM_ADDR_RANGE     ]   dut__tb__sram_result_write_address ,
	output wire signed [`SRAM_DATA_RANGE     ]   dut__tb__sram_result_write_data    ,
	output wire        [`SRAM_ADDR_RANGE     ]   dut__tb__sram_result_read_address  , 
	input  wire signed [`SRAM_DATA_RANGE     ]   tb__dut__sram_result_read_data     ,        

	//scratchpad SRAM interface
	output wire                                 dut__tb__sram_scratchpad_write_enable  ,
	output wire       [`SRAM_ADDR_RANGE     ]   dut__tb__sram_scratchpad_write_address ,
	output wire signed[`SRAM_DATA_RANGE     ]   dut__tb__sram_scratchpad_write_data    ,
	output wire       [`SRAM_ADDR_RANGE     ]   dut__tb__sram_scratchpad_read_address  , 
	input  wire signed[`SRAM_DATA_RANGE     ]   tb__dut__sram_scratchpad_read_data  

	);

	//q_state_output SRAM interface
	reg        [15:0]                sram_write_address_r;
	reg        [15:0]                sram_write_address_sp;
	reg        [15:0]                sram_read_address_I;
	reg        [15:0]                sram_read_address_W;

	reg        [15:0]                sram_read_address_sp;     //Q
	reg        [15:0]                sram_read_address_result; // Read K

	reg signed [`SRAM_DATA_RANGE]    input_read_data;          // 32-bits
	reg signed [`SRAM_DATA_RANGE]    weight_read_data;         // 32-bits
	reg signed [`SRAM_DATA_RANGE]    accum_result;             // C Result from DW_fp_mac
	reg signed [`SRAM_DATA_RANGE]    mac_result_z;
	reg                              compute_complete;
	// Use define to parameterize the variable sizes
	`ifndef SRAM_ADDR_WIDTH
	`define SRAM_ADDR_WIDTH 16
	`endif

	`ifndef SRAM_DATA_WIDTH
	`define SRAM_DATA_WIDTH 32
	`endif

	parameter [4:0]
		S0        = 5'b00000, // Initialization
		S1        = 5'b00001, // Read matrix dimensions
		S2        = 5'b00010, // Update counters
		S3        = 5'b00011, // Read from SRAMs
		S4        = 5'b00100, // Accumulate partial products
		S5        = 5'b00101, // Write to result SRAM
		S5_delay  = 5'b00110, // Q calculations done
		S6        = 5'b00111, // K calculations start
		S7        = 5'b01000, 
		S8        = 5'b01001,
		S8_delay  = 5'b01010, // K calculations end
		S9        = 5'b01011, // V calculations start
		S10       = 5'b01100,
		S11       = 5'b01101,
		S11_delay = 5'b01110, // V calculations end
		S12       = 5'b01111, // S calculations start 
		S13       = 5'b10000,
		S14       = 5'b10001,
		S15       = 5'b10010,
		S15_delay = 5'b10011, // S calculations end
		S16       = 5'b10100, // Completion state
		S17       = 5'b10101, // Z matrix start
		S18       = 5'b10110,
		S19       = 5'b10111,
		S19_delay = 5'b11000,
		S20       = 5'b11001; // End of Z matrix

		reg [4:0] current_state, next_state;

		// Local control path variables
		reg                  set_dut_ready;
		reg [1:0]            input_read_addr_sel;
		reg [1:0]            weight_read_addr_sel;
		reg [1:0]            temp_count_var_I_count_select; // for increasing A until Arow
		reg [1:0]            temp_count_z_select;
		reg [1:0]            write_enable_sel;
		reg                  Q_done, K_done, V_done;
		reg                  Q_K_read, S_V_read;            // Flags for reading Q,K matrices and S,V matrices 
		reg                  S_done, Z_done;
		
		// Local data path variables 
		reg [31:0]           temp_count_var_I_count;        // if equals to brows then start from A1
		reg [15:0]           counter_for_V;
		reg [15:0]           counter_for_S;
		reg [15:0]           rows_I, column_I;
		reg [15:0]           rows_W, column_W;
		//rows_C = rows_I, column_C = column_W;
		reg [15:0]           input_read_address, weight_read_address;
		reg [15:0]           start_address_for_V;
		reg [1:0]            write_enable_sel_r;
		
	//------------------------------------------Address Counters-----------------------------------------------------
	// This temp variable is to track A, i.e if calculation of a C is done then start from 1,
	always @(posedge clk) begin 
		if (!reset_n)
			temp_count_var_I_count <= 32'b0;
		else begin
			case (temp_count_var_I_count_select)
				2'b00: temp_count_var_I_count <= `SRAM_ADDR_WIDTH'b0;
				2'b01: temp_count_var_I_count <= temp_count_var_I_count + `SRAM_ADDR_WIDTH'b1;
				2'b10: temp_count_var_I_count <= temp_count_var_I_count;
				2'b11: temp_count_var_I_count <= `SRAM_ADDR_WIDTH'b01; // when state S5
			endcase
		end
	end

	// another counter for setting where z = a*b +c. If it reaches final value i.e Brows then Write_C then next C calculation -> S2
	// when temp_a == rowB then write to C
	always @(posedge clk) begin
		if (!reset_n) begin
			accum_result <= 32'b0;
		end
		else begin
			if (Q_K_read == 1'b0) begin
				case (temp_count_z_select)
					2'b00: begin
						if (temp_count_var_I_count == 2'b0 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= 32'b0;
					end
					2'b01: begin
						if (temp_count_var_I_count == 2'b0 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= mac_result_z;
					end
					2'b10: begin
						if (temp_count_var_I_count == 0 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= (write_enable_sel_r == 2'b1) ? 32'b0 : accum_result;
					end
				endcase
			end
			else if (Q_K_read == 1'b1 && S_V_read == 1'b0) begin
				case (temp_count_z_select)
					2'b00: begin
						if (temp_count_var_I_count == 0 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= 32'b0;
					end
					2'b01: begin
						if (temp_count_var_I_count == 32'b0 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= mac_result_z;
					end
					2'b10: begin
						if (temp_count_var_I_count == 32'b0 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= (write_enable_sel_r == 2'b1) ? 32'b0 : accum_result;
					end
				endcase
			end
			else if (Q_K_read == 1'b1 && S_V_read == 1'b1) begin
				case (temp_count_z_select)
					2'b00: begin
						if (counter_for_V == 1 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= 32'b0;
					end
					2'b01: begin
						if (counter_for_V == 1 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= mac_result_z;
					end
					2'b10: begin
						if (counter_for_V == 1 && write_enable_sel == 2'b1)
							accum_result <= 32'b0;
						else
							accum_result <= (write_enable_sel_r == 2'b1) ? 32'b0 : accum_result;
					end
				endcase
			end
		end
	end

	//------------------------------------------Addresses calculation -----------------------------------------------------
	// SRAM I read address calculation
	always @(posedge clk) begin
		if (!reset_n) begin
			sram_read_address_I <= `SRAM_ADDR_WIDTH'b0;
			sram_read_address_sp <= `SRAM_ADDR_WIDTH'b0;
			counter_for_S <= 16'b1;
		end
		else begin
			if (Q_K_read == 1'b0 && S_V_read == 1'b0) begin
				case (input_read_addr_sel)
					2'b00: sram_read_address_I <= `SRAM_ADDR_WIDTH'b0;
					2'b01: begin 
						if ((sram_read_address_I == rows_I * column_I) && 
							(sram_write_address_r == column_W*rows_I - 1 || 
							sram_write_address_r == column_W*rows_I*2 - 1 || 
							sram_write_address_r == column_W*rows_I*3 - 1))    
							sram_read_address_I <= `SRAM_ADDR_WIDTH'b1; 
						else
							sram_read_address_I <= sram_read_address_I + `SRAM_ADDR_WIDTH'b1;
					end
					2'b10: sram_read_address_I <= sram_read_address_I;
					2'b11: sram_read_address_I <= sram_read_address_I - (column_I - `SRAM_ADDR_WIDTH'b01);
				endcase
			end
			else if (Q_K_read == 1'b1 && S_V_read == 1'b0) begin // Reading for S
				case (input_read_addr_sel)
					2'b00: sram_read_address_sp <= `SRAM_ADDR_WIDTH'b0;
					2'b01: begin 
						if ((sram_read_address_sp == rows_I * column_W - 1) && 
							(sram_write_address_r == column_W*rows_I*3 + (rows_I*rows_I) - 1))
							sram_read_address_sp <= `SRAM_ADDR_WIDTH'b0; 
						else 
							sram_read_address_sp <= sram_read_address_sp + `SRAM_ADDR_WIDTH'b1;
					end
					2'b10: sram_read_address_sp <= sram_read_address_sp;
					2'b11: sram_read_address_sp <= sram_read_address_sp - (column_W - `SRAM_ADDR_WIDTH'b01);
				endcase
			end
			else if (S_V_read == 1'b1) begin
				case (input_read_addr_sel)
					2'b00: sram_read_address_sp <= column_W*rows_I*3;
					2'b01: begin 
						if (sram_read_address_sp == rows_I*rows_I+column_W*rows_I*3-1 && counter_for_S < column_W) begin
							sram_read_address_sp <= sram_read_address_sp - rows_I + 16'b1;
							counter_for_S <= counter_for_S + 16'b1;
						end
						else if (temp_count_var_I_count == rows_I-1 && counter_for_S == column_W) begin
							sram_read_address_sp <= sram_read_address_sp + 16'b1;
							counter_for_S <= 16'b1;
						end 
						else if (temp_count_var_I_count == rows_I-1 && counter_for_S < column_W) begin
							sram_read_address_sp <= sram_read_address_sp - rows_I + 16'b1;
							counter_for_S <= counter_for_S + 16'b1;
						end
						else begin
							sram_read_address_sp <= sram_read_address_sp + `SRAM_ADDR_WIDTH'b1;
							counter_for_S <= counter_for_S;
						end
					end
					2'b10: sram_read_address_sp <= sram_read_address_sp;
					2'b11: sram_read_address_sp <= sram_read_address_sp - (rows_I - `SRAM_ADDR_WIDTH'b01);
				endcase
			end
		end
	end

	// SRAM W read address calculation
	always @(posedge clk) begin
		if (!reset_n) begin
			sram_read_address_W <= 16'b0;
			counter_for_V <= 16'b1;
			start_address_for_V <= column_W*rows_I*2;
			sram_read_address_result <= column_W * rows_I;
		end
		else begin
			if (Q_K_read == 1'b0 && S_V_read == 1'b0) begin
				case (weight_read_addr_sel)
					2'b00: sram_read_address_W <= `SRAM_ADDR_WIDTH'b0;
					2'b01: begin
						if (Q_done == 1'b0) begin
							sram_read_address_W <= (sram_read_address_W == column_W * rows_W) ? 
												`SRAM_ADDR_WIDTH'b01 : sram_read_address_W + `SRAM_ADDR_WIDTH'b1;
						end
						else if (K_done == 1'b0 && Q_done == 1'b1) begin
							sram_read_address_W <= (sram_read_address_W == column_W * rows_W * 2) ? 
												column_W * rows_W + 16'b1 : sram_read_address_W + `SRAM_ADDR_WIDTH'b1;
						end
						else if (V_done == 1'b0 && K_done == 1'b1 && Q_done == 1'b1) begin
							sram_read_address_W <= (sram_read_address_W == column_W * rows_W * 3) ? 
												column_W * rows_W * 2 + 16'b1 : sram_read_address_W + `SRAM_ADDR_WIDTH'b1;
						end
					end
					2'b10: sram_read_address_W <= sram_read_address_W;
					2'b11: sram_read_address_W <= `SRAM_ADDR_WIDTH'b01;
				endcase
			end
			else if (Q_K_read == 1'b1 && S_V_read == 1'b0) begin
				case (weight_read_addr_sel)
					2'b00: sram_read_address_result <= column_W * rows_I;
					2'b01: sram_read_address_result <= (sram_read_address_result == column_W * rows_I * 2 - 16'b1) ? 
													column_W * rows_I : sram_read_address_result + `SRAM_ADDR_WIDTH'b1;
					2'b10: sram_read_address_result <= sram_read_address_result;
					2'b11: sram_read_address_result <= column_W * rows_I;
				endcase
			end
			else if (S_V_read == 1'b1) begin
				case (weight_read_addr_sel)
					2'b00: begin
						sram_read_address_result <= column_W * rows_I * 2;
						counter_for_V <= 16'b1;
						start_address_for_V <= column_W*rows_I*2;
					end
					2'b01: begin
						if (sram_read_address_result == column_W*rows_I*3 - 1 && counter_for_V == column_W) begin
							sram_read_address_result <= column_W * rows_I * 2;
							start_address_for_V <= column_W*rows_I*2;
							counter_for_V <= 16'b1;
						end
						else if (sram_read_address_result == column_W*rows_I*3 - 1) begin
							sram_read_address_result <= column_W * rows_I * 2;
							start_address_for_V <= column_W*rows_I*2;
							counter_for_V <= 16'b1;
						end
						else if (counter_for_V == rows_I) begin
							sram_read_address_result <= start_address_for_V + 16'b1;
							start_address_for_V <= start_address_for_V + 16'b1; 
							counter_for_V <= 16'b1;
						end
						else begin
							sram_read_address_result <= sram_read_address_result + column_W;
							counter_for_V <= counter_for_V + 16'b1;
						end
					end
					2'b10: sram_read_address_result <= sram_read_address_result;
					2'b11: sram_read_address_result <= column_W * rows_I * 2;
				endcase
			end
		end
	end

	// for result write addr calculation
	always @(posedge clk) begin
		if (!reset_n) begin
			sram_write_address_r <= 16'hFFFF;  
		end
		else begin
			case (write_enable_sel)
				2'b01: sram_write_address_r <= sram_write_address_r + 16'b1;
				2'b00: sram_write_address_r <= sram_write_address_r;
				2'b10: sram_write_address_r <= 16'hFFFF;
			endcase
		end
	end

	//---------------------------------------Next state logic---------------------------------------
	always @(posedge clk) begin
		if (!reset_n)
			current_state <= S0;
		else
			current_state <= next_state;
	end

	//---------------------------------------Current state--------------------------------------------
	always @(*) begin
		case (current_state) //prevent unwanted latches
			S0: begin
				if (dut_valid) begin
					rows_I                          = 16'b0;
					rows_W                          = 16'b0;
					column_I                        = 16'b0;
					column_W                        = 16'b0;
					input_read_addr_sel             = 2'b0;
					set_dut_ready                   = 1'b0;
					weight_read_addr_sel            = 2'b0;
					write_enable_sel                = 2'b0;
					temp_count_var_I_count_select   = 2'b0;
					temp_count_z_select             = 2'b0;
					Q_done                          = 1'b0;
					K_done                          = 1'b0;
					V_done                          = 1'b0;
					Q_K_read                        = 1'b0;
					S_V_read                        = 1'b0;
					next_state                      = S1; 
				end
				else begin 
					next_state                      = S0;
					set_dut_ready                   = 1'b1;
				end
			end

			S1: begin //get rows and col of A,B,C
				set_dut_ready                       = 1'b0;
				input_read_data                     = tb__dut__sram_input_read_data;
				weight_read_data                    = tb__dut__sram_weight_read_data;
				rows_I                              = input_read_data[31:16];
				column_I                            = input_read_data[15:0];
				rows_W                              = weight_read_data[31:16];
				column_W                            = weight_read_data[15:0];
				Q_K_read                            = 1'b0;
				temp_count_var_I_count_select       = 2'b00;
				temp_count_z_select                 = 2'b0;
				input_read_addr_sel                 = 2'b0;
				weight_read_addr_sel                = 2'b0;
				write_enable_sel                    = 2'b0;
				next_state                          = S2;
			end

			S2: begin // update counters only
				set_dut_ready                       = 1'b0;
				temp_count_var_I_count_select       = 2'b10;
				temp_count_z_select                 = 2'b0;
				input_read_addr_sel                 = 2'b01;
				weight_read_addr_sel                = 2'b01;
				write_enable_sel                    = 2'b0;
				Q_K_read                            = 1'b0;
				next_state                          = S3;
			end

			S3: begin // read from SRAM
				set_dut_ready                       = 1'b0;
				temp_count_var_I_count_select       = 2'b01;
				input_read_addr_sel                 = 2'b01;
				weight_read_addr_sel                = 2'b01;
				temp_count_z_select                 = 2'b10;
				write_enable_sel                    = 2'b0;
				Q_K_read                            = 1'b0;
				next_state                          = S4;
			end
			
			S4: begin //Accumulate i.e calculate partial products
				set_dut_ready                       = 1'b0;
				temp_count_var_I_count_select       = 2'b10;
				input_read_addr_sel                 = 2'b10;
				weight_read_addr_sel                = 2'b10;
				temp_count_z_select                 = 2'b01;
			
				if (temp_count_var_I_count + 1 == rows_W) begin
					write_enable_sel                = 2'b1;
					next_state                      = S5;
				end
				else begin
					write_enable_sel                = 2'b0;
					next_state                      = S3;
				end
			end
			
			S5: begin // write into sram c
				set_dut_ready                       = 1'b0;
				write_enable_sel                    = 2'b0;
				if (sram_write_address_r == (column_W*rows_I - 16'b1)) begin
					input_read_addr_sel             = 2'b01;
					weight_read_addr_sel            = 2'b01;
					Q_done                          = 1'b1;
					next_state                      = S5_delay;
				end
				else if (sram_read_address_W == rows_W*column_W) begin //when 1 C calculation is done, start A and B from start
					input_read_addr_sel             = 2'b01;
					weight_read_addr_sel            = 2'b01;
					temp_count_var_I_count_select   = 2'b0;
					temp_count_z_select             = 2'b0;
					next_state                      = S5_delay;
				end  
				else begin // in same row
					input_read_addr_sel             = 2'b11;
					weight_read_addr_sel            = 2'b01;
					temp_count_var_I_count_select   = 2'b0;
					temp_count_z_select             = 2'b0;
					next_state                      = S5_delay;
				end
			end
			
			S5_delay: begin
				input_read_addr_sel                 = 2'b10;
				weight_read_addr_sel                = 2'b10;
				next_state                          = S3;
			
				if (sram_write_address_r == (column_W*rows_I - 16'b1)) begin
					write_enable_sel                = 2'b0;
					input_read_addr_sel             = 2'b10;
					weight_read_addr_sel            = 2'b10;
					temp_count_var_I_count_select   = 2'b0;
					temp_count_z_select             = 2'b0;
					Q_done                          = 1'b1;
					next_state                      = S6;
				end
				else begin
					next_state                      = S3;
				end
			end

			S6: begin //like S3
				// Now for K, Increase write addr, New W will be from rows_W*Col_w + 1 to rows_W*Col_w*2, I same i.e from 1
				set_dut_ready                   = 1'b0;
				temp_count_var_I_count_select   = 2'b01;
				input_read_addr_sel             = 2'b01;
				weight_read_addr_sel            = 2'b01;
				temp_count_z_select             = 2'b10;
				write_enable_sel                = 2'b0;
				next_state                      = S7;
			end
			
			S7: begin //like s4  //Accumulate i.e calculate partial products
				set_dut_ready                   = 1'b0;
				temp_count_var_I_count_select   = 2'b10;  //Accumulated now count 
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10;
				temp_count_z_select             = 2'b01;
			
				if (temp_count_var_I_count + 16'b1 == rows_W) begin
					write_enable_sel            = 2'b1;
					next_state                  = S8;
				end
				else begin
					write_enable_sel            = 2'b0;
					next_state                  = S6;
				end
			end
			
			S8: begin //like S5
				set_dut_ready                   = 1'b0;
				write_enable_sel                = 2'b0;
				if (sram_write_address_r == (column_W*rows_I*2 - 16'b1)) begin
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					K_done                      = 1'b1;
					next_state                  = S8_delay;
				end
				else if (sram_read_address_W == rows_W*column_W*2) begin //when 1 C calculation is done, start A and B from start
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select= 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S8_delay;
				end         
				else begin // in same row
					input_read_addr_sel         = 2'b11;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select= 2'b0;					
					temp_count_z_select         = 2'b0;
					next_state                  = S8_delay; // hold state
				end
			end
			
			S8_delay: begin
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10;
				next_state                      = S6;
			
				if (sram_write_address_r == (column_W*rows_I*2 - 16'b1)) begin
					write_enable_sel            = 2'b0;
					input_read_addr_sel         = 2'b10;
					weight_read_addr_sel        = 2'b10; 
				temp_count_var_I_count_select= 2'b0;
					temp_count_z_select         = 2'b0;
					K_done                      = 1'b1;
					next_state                  = S9;
				end
				else 
					next_state                  = S6;
			end
			
			S9: begin // S3,S6
				// Now for V, Increase write addr, New W will be from rows_W*Col_w*2 + 1 to rows_W*Col_w*3, I same i.e from 1
				set_dut_ready                   = 1'b0;
				temp_count_var_I_count_select   = 2'b01;
				input_read_addr_sel             = 2'b01;
				weight_read_addr_sel            = 2'b01;
				temp_count_z_select             = 2'b10;
				write_enable_sel                = 2'b0;
				next_state                      = S10;
			end
			
			S10: begin //Accumulate i.e calculate partial products
				set_dut_ready                   = 1'b0;
				temp_count_var_I_count_select   = 2'b10;  //Accumulated now count 
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10;
				temp_count_z_select             = 2'b01;
			
				if (temp_count_var_I_count + 16'b1 == rows_W) begin
					write_enable_sel            = 2'b1;
					next_state                  = S11;
				end
				else begin
					write_enable_sel            = 2'b0;
					next_state                  = S9;
				end
			end
			
			S11: begin // Write V into SRAM
				set_dut_ready                   = 1'b0;
				write_enable_sel                = 2'b0;
				if (sram_write_address_r == (column_W*rows_I*3 - 16'b1)) begin
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					V_done                      = 1'b1;
					next_state                  = S11_delay;
				end
				else if (sram_read_address_W == rows_W*column_W*3) begin //when 1 C calculation is done, start A and B from start
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S11_delay;
				end         
				else begin // in same row
					input_read_addr_sel         = 2'b11;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S11_delay; // hold state
				end
			end
			
			S11_delay: begin
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10;
				next_state                      = S9;
				if (sram_write_address_r == (column_W*rows_I*3 - 16'b1)) begin
					write_enable_sel            = 2'b0;
					input_read_addr_sel         = 2'b00; // 10
					weight_read_addr_sel        = 2'b00; // 10
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					V_done                      = 1'b1;
					Q_K_read                    = 1'b1;
					next_state                  = S12;
				end
				else 
					next_state                  = S9;
			end
			
			S12: begin // S2, make all counters/select lines = 0, except var_I count
				set_dut_ready                   = 1'b0;
				input_read_addr_sel             = 2'b00;
				weight_read_addr_sel            = 2'b00;
				write_enable_sel                = 2'b00;
				temp_count_var_I_count_select   = 2'b00;
				temp_count_z_select             = 2'b00;
				next_state                      = S13;
			end
			
			S13: begin // S3
				set_dut_ready                   = 1'b0;
				input_read_addr_sel             = 2'b01;
				weight_read_addr_sel            = 2'b01;
				write_enable_sel                = 2'b00;
				temp_count_var_I_count_select   = 2'b01;
				temp_count_z_select             = 2'b10;
				if (temp_count_var_I_count + 16'b1 == column_W) begin
					write_enable_sel            = 2'b1;
					next_state                  = S15;
				end
				next_state                      = S14;
			end
			
			S14: begin //S4 //Accumulate i.e calculate partial products
				set_dut_ready                   = 1'b0;
				temp_count_var_I_count_select   = 2'b10;
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10; //10
				temp_count_z_select             = 2'b01;
			
				if (temp_count_var_I_count + 16'b1 == column_W) begin
					write_enable_sel            = 2'b1;
					next_state                  = S15;
				end
				else begin
					write_enable_sel            = 2'b0;
					next_state                  = S13;
				end
			end

			S15: begin // write into sram c
				set_dut_ready                   = 1'b0;
				write_enable_sel                = 2'b0;
				if (sram_write_address_r == (rows_I*rows_I+rows_I*column_W*3 - 16'b1)) begin
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0; // edit
					next_state                  = S15_delay; //S5_delay
				end
				else if (sram_read_address_result == rows_I*column_W*2 - 16'b1) begin //when 1 C calculation is done, start A and B from start
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S15_delay;
				end   
				else begin
					input_read_addr_sel         = 2'b11; //11
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S15_delay; // hold state
				end
			end

			S15_delay: begin
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10;
				next_state                      = S13;
				if (sram_write_address_r == (rows_I*rows_I+rows_I*column_W*3 - 16'b1)) begin
					write_enable_sel            = 2'b0;
					input_read_addr_sel         = 2'b00;
					weight_read_addr_sel        = 2'b00;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					S_done                      = 1'b1;
					S_V_read                    = 1'b1;
					next_state                  = S16;
				end 
			end

			S16: begin // 12
				set_dut_ready                   = 1'b0;
				S_V_read                        = 1'b1;
				input_read_addr_sel             = 2'b00;
				weight_read_addr_sel            = 2'b00;
				write_enable_sel                = 2'b00;
				temp_count_var_I_count_select   = 2'b00;
				temp_count_z_select             = 2'b00;
				next_state                      = S17;
			end 

			S17: begin // 13
				set_dut_ready                   = 1'b0;
				input_read_addr_sel             = 2'b01;
				weight_read_addr_sel            = 2'b01;
				write_enable_sel                = 2'b00;
				temp_count_var_I_count_select   = 2'b01;
				temp_count_z_select             = 2'b10;
				if (counter_for_V == rows_I) begin
					write_enable_sel            = 2'b1;
					input_read_addr_sel         = 2'b10;
					next_state                  = S18;
				end
				else
					next_state                  = S18;
			end
			S18: begin // 14
				set_dut_ready                   = 1'b0;
				temp_count_var_I_count_select   = 2'b10;
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10; //10
				temp_count_z_select             = 2'b01;
			
				// if( temp_count_var_I_count + 1 == column_C) begin
				if (counter_for_V == rows_I) begin
					write_enable_sel            = 2'b1;
					next_state                  = S19;
				end
				else begin
					write_enable_sel            = 2'b0;
					next_state                  = S17;
				end
			end
			
			S19: begin // 15
				set_dut_ready                   = 1'b0;
				write_enable_sel                = 2'b0;
				if (sram_write_address_r == (rows_I*rows_I+rows_I*column_W*4 - 16'b1)) begin
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					Z_done                      = 1'b1;
					temp_count_var_I_count_select = 2'b0; // edit
					next_state                  = S19_delay; //S5_delay
				end
				else if (sram_read_address_result == rows_I*column_W*3 - 16'b1) begin //when 1 C calculation is done, start A and B from start
					input_read_addr_sel         = 2'b01;
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b10;
					temp_count_z_select         = 2'b0;
					next_state                  = S19_delay;
				end 
				else if ((temp_count_var_I_count == rows_I - 32'b1) && (counter_for_S == column_W)) begin
					input_read_addr_sel         = 2'b01; //01 //edit 189
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S19_delay; // hold state
				end
				else begin
					input_read_addr_sel         = 2'b01; //01 //edit 189
					weight_read_addr_sel        = 2'b01;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					next_state                  = S19_delay; // hold state
				end
			end
			
			S19_delay: begin // 15_delay
				input_read_addr_sel             = 2'b10;
				weight_read_addr_sel            = 2'b10;
				next_state                      = S17;
				temp_count_var_I_count_select   = 2'b0;
				temp_count_z_select             = 2'b10;
				if (sram_write_address_r == (rows_I*rows_I+rows_I*column_W*4 - 16'b1)) begin
					write_enable_sel            = 2'b0;
					input_read_addr_sel         = 2'b00;
					weight_read_addr_sel        = 2'b00;
					temp_count_var_I_count_select = 2'b0;
					temp_count_z_select         = 2'b0;
					Z_done                      = 1'b1;
					S_V_read                    = 1'b0;
					next_state                  = S20;
				end 
			end
			
			S20: begin // 16
				set_dut_ready                   = 1'b1;
				rows_I                          = 16'b0;
				rows_W                          = 16'b0;
				column_I                        = 16'b0;
				column_W                        = 16'b0;
				// column_C                     = 16'b0;
				input_read_addr_sel             = 2'b0;
				set_dut_ready                   = 1'b0;
				weight_read_addr_sel            = 2'b0;
				write_enable_sel                = 2'b10;
				temp_count_var_I_count_select   = 2'b0;
				temp_count_z_select             = 2'b0;
				Q_done                          = 1'b0;
				K_done                          = 1'b0;
				V_done                          = 1'b0;
				Z_done                          = 1'b0;
				next_state                      = S0;
			end
			
			default: begin 
				next_state                      = S0;
			end
		endcase
	end

	always@(posedge clk) begin
		write_enable_sel_r <= write_enable_sel;
	end

	assign dut__tb__sram_result_write_data       = mac_result_z;
	assign dut__tb__sram_scratchpad_write_data   = mac_result_z;
	assign dut__tb__sram_input_read_address      = Q_K_read? 0: sram_read_address_I;
	assign dut__tb__sram_weight_read_address     = Q_K_read? 0:  sram_read_address_W;

	assign dut__tb__sram_result_write_address    = sram_write_address_r;
	assign dut__tb__sram_scratchpad_write_address = sram_write_address_r;
	assign dut__tb__sram_result_write_enable     = write_enable_sel_r;
	assign dut__tb__sram_scratchpad_write_enable = write_enable_sel_r;

	assign dut__tb__sram_input_write_enable  = 1'b0;
	assign dut__tb__sram_input_write_address = 16'b0;
	assign dut__tb__sram_input_write_data    = 32'b0;
	assign dut__tb__sram_weight_write_enable = 1'b0;
	assign dut__tb__sram_weight_write_address= 16'b0;
	assign dut__tb__sram_weight_write_data   = 32'b0;

	assign dut__tb__sram_result_read_address = sram_read_address_result;
	assign dut__tb__sram_scratchpad_read_address = sram_read_address_sp;
		
	// DUT ready handshake logic
	always @(posedge clk) begin : proc_compute_complete
		if(!reset_n) begin
			compute_complete <= 1'b0;
		end
		else begin
			compute_complete <= (set_dut_ready == 1'b1) ? 1'b1 : 1'b0;
		end
	end

	assign dut_ready = compute_complete;


	INT_MUL
	MUL_Inst(
	.inst_a(dut_valid? 32'b0 :(Q_K_read? tb__dut__sram_scratchpad_read_data: tb__dut__sram_input_read_data)),
	.inst_b(dut_valid? 1'b0 :(Q_K_read? tb__dut__sram_result_read_data: tb__dut__sram_weight_read_data)),
	.inst_c(accum_result),
	.z_inst(mac_result_z)
	);

endmodule: MyDesign

module INT_MUL (
    input wire  signed [31:0]  inst_a,
    input wire  signed [31:0]  inst_b,
    input wire  signed [31:0]  inst_c,
    output wire signed [31:0] z_inst
  );
  
    // Simple multiplication
    wire signed [31:0] mult_result;
    assign mult_result = inst_a * inst_b;
  
    // Add the multiplication result to inst_c
    assign z_inst = mult_result + inst_c;
  
endmodule: INT_MUL
