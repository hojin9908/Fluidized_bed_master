

void PCISPH(Csolver solv)
{
	{
		//-------------------------------------------------------------------------------------------------
		// ##. GPU device properties
		//-------------------------------------------------------------------------------------------------

		cudaDeviceProp prop;
		{
			int_t count;

			cudaGetDeviceCount(&count);

			for (int_t i = 0; i < count; i++)
			{
				cudaGetDeviceProperties(&prop, i);

				cout << "### GPU DEVICE PROPERTIES................................." << endl << endl;
				cout << "	Name: " << prop.name << endl;
				cout << "	Compute capability: " << prop.major << "." << prop.minor << endl;
				cout << "	Clock rate: " << prop.clockRate << endl;
				cout << "	Total global memory: " << prop.totalGlobalMem << endl;
				cout << "	Total constant memory: " << prop.memPitch << endl;
				cout << "	Multiprocessor count: " << prop.multiProcessorCount << endl;
				cout << "	Shared mem per block: " << prop.sharedMemPerBlock << endl;
				cout << "	Registers per block: " << prop.regsPerBlock << endl;
				cout << "	Threads in warp: " << prop.warpSize << endl;
				cout << "	Max threads per block: " << prop.maxThreadsPerBlock << endl;
				cout << "	Max thread dimensions: " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << endl;
				cout << "	Max grid dimensions: " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << endl;
				cout << "..........................................................." << endl;
				cout << endl;
			}
		}


		// print ----
		cout << " ---------------------------" << endl;
		cout << " SOPHIA_gpu v.1.0 " << endl;
		cout << " Developed by E.S. Kim, Y.B. Cho, S.H. Park" << endl;
		cout << " 2017. 02. 20 " << endl;
		cout << "----------------------------" << endl;
		cout << endl;
		//-------------------






		//-------------------------------------------------------------------------------------------------
		// ##. READ SOLV & DECLARE VARIABLES
		//-------------------------------------------------------------------------------------------------

		// solv variable

		char* INPUT_FILE_NAME = "./input/p_type.txt";							// input file name and address
		char* OUTPUT_FILE_NAME = "./output/output.txt";							// output file name and address


		// solution variables
		Real kappa = solv.kappa;												// k in k*h
		int_t dim = solv.dim;													// dimension
		int_t pnb_size = solv.pnb_size;
		int_t kernel_type = solv.kernel_type;									// kernel type
		int_t flt_type = solv.flt_type;											// filter type
		int_t rho_type = solv.rho_type;											// density calcuation type
		int_t time_type = solv.time_type;										// time stepping type
		int_t fluid_type = solv.fluid_type;										// fluid type
		int_t simulation_type = solv.simulation_type;							// simulation type (single_phase / two_phase)
		int_t neighbor_cell_type = solv.nb_cell_type;							// neighbor cell type (0: 3x3, 1: 5x5)
		int_t nb_cell_number = 3;												// 3: 3x3, 5: 5x5						


		Real p_ref = solv.p_ref;												// reference pressure (for EOS)
		Real gamma = solv.gamma;												// gamma
		Real dt = solv.dt;														// time-step(s)
		Real dt0 = dt;
		Real time = solv.time;													// time(s)
		Real time_end = solv.time_end;
		int_t count = floor(solv.time / solv.dt + 0.5);							// starting number of count
		Real Xmargin_m = solv.Xmargin_m, Xmargin_p = solv.Xmargin_p,
			Ymargin_m = solv.Ymargin_m, Ymargin_p = solv.Ymargin_p,
			Zmargin_m = solv.Zmargin_m, Zmargin_p = solv.Zmargin_p;				// margin for simulation range
		Real x_min, x_max, y_min, y_max, z_min, z_max;
		
		Real C_xsph = solv.c_xsph;												// coefficient for XSPH
		Real C_repulsive = solv.c_repulsive;									// coefficient for Repulsive boundary force
		Real u_limit = solv.u_limit;											// velocity limit		


		Real cell_reduction_factor = 1.1, search_incr_factor = 1.1;				// coefficient for cell and search range (esk)


		int_t freq_cell = solv.freq_cell;										// cell initialization frequency								
		int_t freq_filt = solv.freq_filt;										// filtering frequency
		int_t freq_mass_sum = solv.freq_mass_sum;								// mass summation frequency
		int_t freq_temp = solv.freq_temp;										// temperature filtering frequency
		int_t freq_output = solv.freq_output;									// output frequency
		int_t freq_neighbor_search = solv.freq_cell;							// neighbor search frequency (esk)


		int_t fp_solve = solv.fp_solve;
		int_t fv_solve = solv.fv_solve;
		int_t fva_solve = solv.fva_solve;
		int_t fg_solve = solv.fg_solve;
		int_t fs_solve = solv.fs_solve;
		int_t surf_model = solv.surf_model;
		int_t interface_solve = solv.interface_solve;
		int_t fb_solve = solv.fb_solve;
		int_t con_solve = solv.con_solve;
		int_t boussinesq_solve = solv.boussinesq_solve;

		int_t kgc_solve = solv.kgc_solve;										// kernel gradient correction
		int_t delSPH_solve = solv.delSPH_solve;									// delta SPH for weakly compressible
		int_t delSPH_model = solv.delSPH_model;									// delta SPH model
		int_t pst_solve = solv.pst_solve;										// particle shifting technique

		int_t flag_z_indexing = solv.flag_z_index;									// z-indexing (on/off)	
		int_t flag_timestep_update = solv.flag_timestep_update;						// flag for varying timestep update

		//-----------------------------------------------------------------------------------------------------
		// solution variable for PCISPH / DFSPH
		//-----------------------------------------------------------------------------------------------------
		int_t xsph_solve = solv.xsph_solve;			// solve xsph ? 
		Real drho_th = solv.drho_th;				// density convergence criterion
		Real dp_th = solv.dp_th;					// pressure convergence criterion
		Real p_relaxation = solv.p_relaxation;			// relaxation factor for PCISPH pressure 
		int_t minIteration = solv.minIteration;			// minimum number of PCISPH iteration
		int_t maxIteration = solv.maxIteration;			// maximum number of PCISPH iteration


		int_t number_of_particles = gpu_count_particle_numbers(INPUT_FILE_NAME);	// calculating number of particles (from input file)
		int_t number_of_boundaries = gpu_count_boundary_numbers(INPUT_FILE_NAME);	// calculating number of bondary particles (from input file)


		int_t particle_block_size = ceil(sqrt(number_of_particles));				// block size for 2-dimensional block 
		dim3 particle_blocks(particle_block_size, particle_block_size);				// 2-dimensional block


		if (neighbor_cell_type == 1)
		{
			cell_reduction_factor = 0.7;
			search_incr_factor = 1.2;
			nb_cell_number = 5;
		}




		//-------------------------------------------------------------------------------------------------
		// ##. GENERATE PARTICLES
		//-------------------------------------------------------------------------------------------------

		// declare
		Cuda_Particle_Array host_particle_array(number_of_particles, number_of_boundaries, pnb_size);
		Cuda_Particle_Array	particle_array(number_of_particles, number_of_boundaries, pnb_size);
		Cuda_Particle_Array particle_array_temp(number_of_particles, number_of_boundaries, pnb_size);

		// read Csolver input file (in cpu, gpu)
		host_particle_array.read_solv(solv);
		particle_array.read_solv(solv);

		// memory allocation
		host_particle_array.host_allocate();
		particle_array.device_allocate();
		if (flag_z_indexing == 1) particle_array_temp.device_allocate();

		// read input file (in gpu)
		host_particle_array.read_input();
		if (flag_z_indexing == 1) particle_array_temp.read_solv(solv);


		// copy host to device (particle_array)
		// particle_array.copy_from_host(host_particle_array);


		// set up proves
		thrust::device_ptr<Real> thrust_umag_ptr = thrust::device_pointer_cast(particle_array.ux);
		thrust::device_ptr<Real> thrust_ftotal_ptr = thrust::device_pointer_cast(particle_array.ftotal);
		thrust::device_ptr<Real> thrust_stiffness = thrust::device_pointer_cast(particle_array.stiffness);
		thrust::device_ptr<Real> thrust_rho_err = thrust::device_pointer_cast(particle_array.rho_err);

		// proving variables:max_umag, max_ftotal
		Real max_umag, max_ftotal;
		//-----------------Added Variations--------------------------------------------------------------//
		Real max_stiffness;
		Real max_rho_err;
		int iteration;
		Real reference_density = 1000.;



		//-------------------------------------------------------------------------------------------------
		// ##. GENERATE CELLS
		//-------------------------------------------------------------------------------------------------

		// cell variables______

		// cell size: NI, NJ, NK
		int_t NI, NJ, NK;

		// number of cells
		int_t number_of_cells;

		// kernel distance: h
		Real KERNEL_DISTANCE;

		// neighbor cell list
		neighbor_cell nb_cell_list;

		// kernel distance: h
		KERNEL_DISTANCE = host_particle_array.h[0];

		// cell range
		Real *host_cell_range, *cell_range;

		host_cell_range = (Real*)malloc(6 * sizeof(Real));		// memory allocation
		cudaMalloc((void **)&cell_range, 6 * sizeof(Real));


		// estimate simulation range
		x_min = host_particle_array.find_xmin(Xmargin_m);	host_cell_range[0] = x_min;
		x_max = host_particle_array.find_xmax(Xmargin_p);	host_cell_range[1] = x_max;
		y_min = host_particle_array.find_ymin(Ymargin_m);	host_cell_range[2] = y_min;
		y_max = host_particle_array.find_ymax(Ymargin_p);	host_cell_range[3] = y_max;
		z_min = host_particle_array.find_zmin(Zmargin_m);	host_cell_range[4] = z_min;
		z_max = host_particle_array.find_zmax(Zmargin_p);	host_cell_range[5] = z_max;


		// estimate number of cells 
		NI = host_particle_array.calculate_NI(cell_reduction_factor*kappa*KERNEL_DISTANCE);
		NJ = host_particle_array.calculate_NJ(cell_reduction_factor*kappa*KERNEL_DISTANCE);
		NK = host_particle_array.calculate_NK(cell_reduction_factor*kappa*KERNEL_DISTANCE);

		number_of_cells = NI * NJ * NK;




		// copy particle_array from host to device
		particle_array.copy_from_host(host_particle_array);
		if (flag_z_indexing == 1) particle_array_temp.copy_from_host(host_particle_array);




		//-------------------------------------------------------------------------------------------------
		// ##. NEIGHBOR SEARCH
		//-------------------------------------------------------------------------------------------------

		// variable
		Cell_Index_Container Cell_Container(number_of_particles, NI, NJ, NK);

		// allocate cell container
		Cell_Container.allocate();
		Cell_Container.initialize(solv);
		Cell_Container.reset_cell_indexing_arrays();

		// z-indexing
		if (flag_z_indexing == 1)
			Cell_Container.make_z_index_map();

		Cell_Container.apply_z_index_map();

		// assingn particles to cell (build PID, CID)
		particle_array.assign_to_cell(Cell_Container.PID, Cell_Container.CID);

		// bin count (build ncell)
		Cell_Container.count_bin();

		// find cumulative sum (build cum_ncell)
		thrust::inclusive_scan(thrust::device, Cell_Container.gpu_ncell, Cell_Container.gpu_ncell + number_of_cells, Cell_Container.gpu_cum_ncell);

		// find start_idx (build start_idx)
		Cell_Container.find_start_idx();

		// sort (build sorted_CID, sorted_PID)
		Cell_Container.count_sort();

		// search particles
		//particle_array.search_neighbors(Cell_Container.nb_cell_list, Cell_Container.sorted_PID, Cell_Container.gpu_ncell, Cell_Container.gpu_start_idx, Cell_Container.g_array_ic_to_iz, search_incr_factor*kappa);
		particle_array.search_neighbors(&Cell_Container);




		//-------------------------------------------------------------------------------------------------
		// ##. ESTIMATE KERNEL & FILTER (INITIAL)
		//-------------------------------------------------------------------------------------------------

		particle_array.calculate_kernel();
		particle_array.calculate_gradient_correction();

		particle_array.calculate_filter();

		if (pst_solve == 1)
		{
			particle_array.calculate_w_dx();
		}

		particle_array.calculate_stiffness(dt);
		max_stiffness = *(thrust::max_element(thrust_stiffness, thrust_stiffness + number_of_particles));
		cout << "Calculated stiffness parameter  :  " << max_stiffness << endl;

		initialize_PCISPH(particle_array, max_stiffness, dt);




		//-------------------------------------------------------------------------------------------------
		// ##. PRINT INITIAL STATUS
		//-------------------------------------------------------------------------------------------------

		// print out status____________
		cout << "-----------------------------------------------------------" << endl;
		cout << "Input Summary: " << endl;
		cout << "-----------------------------------------------------------" << endl;
		cout << "	Total number of particles = " << number_of_particles << endl;
		cout << "	pnb_size = " << pnb_size << endl;
		//cout << "		thread size = " << thread_size << endl << endl;
		cout << "	NI = " << NI << ",	NJ = " << NJ << ",	NK = " << NK << endl;
		cout << "-----------------------------------------------------------" << endl << endl;
		//______________________________


		// Input Check ____________
		cout << "-----------------------------------------------------------" << endl;
		cout << "Input Check: " << endl;
		cout << "-----------------------------------------------------------" << endl;

		if (number_of_particles > NUM_PART)	// check number of particles
		{
			cout << "The number of particles exceeds the maximum particle setup (=" << NUM_PART << ")" << endl;
		}
		else
		{
			cout << "Number of particles are appropriate." << endl;
		}

		if (pnb_size > NB_SIZE)	// check number of neighbors
		{
			cout << "The neighbor particle numbers exceeds the maximum setup (=" << NB_SIZE << ")" << endl;
		}
		else
		{
			cout << "Number of neighbors are appropriate." << endl;
		}

		if (flag_z_indexing == 1)
		{
			cout << "z-indexing is ON" << endl;
		}
		else
		{
			cout << "z-indexing is OFF." << endl;
		}

		cout << "-----------------------------------------------------------" << endl << endl;
		//______________________________

		system("pause");


		// print out status____________
		{
			cout << endl;
			cout << "-----------------------------" << endl;
			cout << "Start Simultion!!" << endl;
			cout << "-----------------------------" << endl;
			cout << endl;
		}


		// check results
		if (false)
		{
			cout << "dim = " << dim << endl;
			cout << "check results" << endl << endl;

			int_t *host_ncell;	host_ncell = (int_t*)malloc(number_of_cells * sizeof(int_t));

			host_particle_array.copy_from_device(particle_array);
			host_particle_array.check_results(100);

			if (true)
			{
				host_particle_array.save_pnb("./result_check/check_pnb_dm.txt", 20);
				host_particle_array.save_dist("./result_check/check_dist_dm.txt", 20);
				host_particle_array.save_Wij("./result_check/wij_dm.txt", 20);
				host_particle_array.save_dWij("./result_check/dwij_dm.txt", 20);
			}

			system("pause");
		}





		//-------------------------------------------------------------------------------------------------
		// ##. CODE MAIN LOOP 
		//-------------------------------------------------------------------------------------------------

		// code main
		try	// Exception handling											
		{

			while (time < time_end)
			{


				//-------------------------------------------------------------------------------------------------
				// ##. NEIGHBOR SEARCH
				//-------------------------------------------------------------------------------------------------

				// prepare cells
				if ((count % freq_cell) == 0)
				{

					// re-indexing
					if (flag_z_indexing == 1)
					{
						particle_array.reindex_to(&particle_array_temp, Cell_Container.sorted_PID);
						particle_array_temp.reindex_to(&particle_array, Cell_Container.PID);
					}


					// reset cells (ncell, start_idx, ...)
					Cell_Container.reset_cell_indexing_arrays();

					// assign particles to cells
					particle_array.assign_to_cell(&Cell_Container);

					// convert cell index to z-index
					if (flag_z_indexing == 1)
						Cell_Container.convert_CID_to_ZID();

					// bin count (build ncell)
					Cell_Container.count_bin();

					// find cumulative sum (build cum_ncell)
					thrust::inclusive_scan(thrust::device, Cell_Container.gpu_ncell, Cell_Container.gpu_ncell + number_of_cells, Cell_Container.gpu_cum_ncell);

					// find start_idx (build start_idx)
					Cell_Container.find_start_idx();

					// sort (build sorted_CID, sorted_PID)
					Cell_Container.count_sort();
				}


				// search neighbors
				if ((count % freq_neighbor_search) == 0)
				{
					//particle_array.search_neighbors(Cell_Container.nb_cell_list, Cell_Container.sorted_PID, Cell_Container.gpu_ncell, Cell_Container.gpu_start_idx, Cell_Container.g_array_ic_to_iz, search_incr_factor*kappa);
					particle_array.search_neighbors(&Cell_Container);
				}
				else
				{
					particle_array.calculate_dist();
				}



				//-------------------------------------------------------------------------------------------------
				// ##. KERNEL UPDATE
				//-------------------------------------------------------------------------------------------------

				particle_array.calculate_kernel();
				//particle_array.calculate_gradient_correction();



				//-------------------------------------------------------------------------------------------------
				// ##. DENSITY UPDATE
				//-------------------------------------------------------------------------------------------------

				particle_array.calculate_density();



				//-------------------------------------------------------------------------------------------------
				// ##. ADVECTION FORCE UPDATE
				//-------------------------------------------------------------------------------------------------

				particle_array.calculate_advforce(dt);




				//-------------------------------------------------------------------------------------------------
				// ##. PREDICTOR (PCISPH)
				//-------------------------------------------------------------------------------------------------

				particle_array.predictor_PCISPH(dt);




				//-------------------------------------------------------------------------------------------------
				// ##. PRESSURE / DENSITY UPDATE
				//-------------------------------------------------------------------------------------------------


				// PCISPH Loop
				iteration = 0;

				for (int k = 0; k < maxIteration; k++)
				{
					iteration++;

					// calculate PCISPH algorithm
					particle_array.calculate_PCISPH(dt, max_stiffness);

					// check density divergence criterion
					max_rho_err = *(thrust::max_element(thrust_rho_err, thrust_rho_err + number_of_particles));

					if (iteration >= minIteration)
					{
						if (max_rho_err < reference_density * drho_th)
						{
							goto outOfLoop;
						}
					}
				}

			outOfLoop:;




				//-------------------------------------------------------------------------------------------------
				// ##. FILTER UPDATE
				//-------------------------------------------------------------------------------------------------

				// filter estimation (update)
				if ((count % freq_cell) == 0)
				{
					particle_array.calculate_filter();
				}




				//-------------------------------------------------------------------------------------------------
				// ##. TIME INTEGRATION: UPDATE PROPERTY (PCISPH)
				//-------------------------------------------------------------------------------------------------

				particle_array.sum_force();
				particle_array.update_properties_PCISPH(dt);


				if (solv.pst_solve == 1)
				{
					//particle_array.calculate_number_density();
					particle_array.calculate_surface_normal();
					particle_array.update_particle_shifting(dt);
				}


				//-------------------------------------------------------------------------------------------------
				// ##. TIME STEP CONTROL & UPDATE
				//-------------------------------------------------------------------------------------------------


				if (flag_timestep_update == 1)
				{
					max_umag = *(thrust::max_element(thrust_umag_ptr, thrust_umag_ptr + number_of_particles));
					max_ftotal = *(thrust::max_element(thrust_ftotal_ptr, thrust_ftotal_ptr + number_of_particles));
				}


				// A time-step control needs to be added.

				// time-step calculation
				// dt = clc_timestep(particle, number_of_particles, dt0);

				// time-step update
				time += dt;
				++count;




				//-------------------------------------------------------------------------------------------------
				// ##. Print Output Files
				//-------------------------------------------------------------------------------------------------		

				if (((count - 1) % freq_output) == 0)
				{
					// copy device to host
					host_particle_array.copy_from_device(particle_array);

					// save *.txt
					//host_particle_array.save_plot_xyz(time - dt, count-1);

					// save *.vtk files
					host_particle_array.save_plot_fluid_vtk(time - dt, count - 1);

					cout << "time = " << time - dt << " [sec]   /   " << "count = " << count -1 << " [step] " << endl;
					//cout << "max_umag = " << max_umag << "\t" << "max_ftotal = " << max_ftotal << endl << endl;
					cout << "Number of PCISPH convergence iterations : " << iteration << endl;
				}

			}




			//-------------------------------------------------------------------------------------------------
			// ##. Save Final Output 
			//-------------------------------------------------------------------------------------------------


			// Save output files
			if (false)
			{
				// Save Files

				// Save Restart File
			}

		}
		catch (const char* ex)	// Exception handling
		{
			cout << "exception handling: " << ex << endl;
			cout << "time = " << time << " [sec]" << endl;
			cout << "saving restart files..." << endl << endl;

			// Save Restart Files
		}




		//-------------------------------------------------------------------------------------------------
		// ##. Memory Release 
		//-------------------------------------------------------------------------------------------------

		free(host_cell_range);
		cudaFree(cell_range);
	}


	//cout << endl << "Memory Leakage (Yes(1):NO(0)): " << _CrtDumpMemoryLeaks() << endl << endl;		// checking memory leaks

	system("pause");


}




