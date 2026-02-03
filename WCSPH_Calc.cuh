//-------------------------------------------------------------------------------------------------
// 모든 입자 계산을 수행하는 주함수
//-------------------------------------------------------------------------------------------------
void SOPHIA_single(int_t*g_idx,int_t*p_idx,int_t*g_idx_in,int_t*p_idx_in,int_t*g_str,int_t*g_end,
									part1*dev_P1,part1*dev_SP1,part2*dev_P2,part2*dev_SP2,part3*dev_P3,
									int_t*p2p_af_in,int_t*p2p_idx_in,int_t*p2p_af,int_t*p2p_idx,
									void*dev_sort_storage,size_t*sort_storage_bytes,part1*file_P1,part2*file_P2,part3*file_P3,int tid)
{

	dim3 b,t;
	t.x=128;
	int s=sizeof(int)*(t.x+1);

	// Eulerian-SPH
	KERNEL_set_alpha<<<b,t>>>(dev_P1);	

	//-------------------------------------------------------------------------------------------------
	// PREDICTOR (Optional)
	//-------------------------------------------------------------------------------------------------

	if(time_type==Pre_Cor){
		b.x=(num_part2-1)/t.x+1;
		KERNEL_clc_predictor<<<b,t>>>(dt,time,dev_P1,dev_P2,dev_P3);
		//KERNEL_clc_predictor_1<<<b,t>>>(count,dt,time,dev_P1,dev_P2,dev_P3);

		cudaDeviceSynchronize();
	}


	//-------------------------------------------------------------------------------------------------
	// 주변입자 검색
	//-------------------------------------------------------------------------------------------------

	if (count==0){		// Eulerian=첫번째 스텝에서만 NNPS
		// g_str을 리셋
		cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells);

		// 입자의 셀번호 계산
		b.x=(num_part2-1)/t.x+1;
		KERNEL_index_particle_to_cell<<<b,t>>>(g_idx_in,p_idx_in,dev_P1);
		cudaDeviceSynchronize();

		// 셀번호를 바탕으로 정렬
		cub::DeviceRadixSort::SortPairs(dev_sort_storage,*sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part2);
		cudaDeviceSynchronize();

		// 정렬한 입자를 재배치
		b.x=(num_part2-1)/t.x+1;
		KERNEL_reorder<<<b,t,s>>>(g_idx,p_idx,g_str,g_end,dev_P1,dev_P2,dev_SP1,dev_SP2);
		cudaDeviceSynchronize();

		// 일부 입자정보 리셋
		cudaMemset(dev_P3,0,sizeof(part3)*num_part2);

		// 입자정보를 P1 에 복사
		cudaMemcpy(dev_P1,dev_SP1,sizeof(part1)*num_part2,cudaMemcpyDeviceToDevice);
		pthread_barrier_wait(&barrier);
	}
	else{
		cudaMemset(dev_P3,0,sizeof(part3)*num_part2);

		// 입자정보를 SP1 에 복사
		cudaMemcpy(dev_SP1,dev_P1,sizeof(part1)*num_part2,cudaMemcpyDeviceToDevice);
		pthread_barrier_wait(&barrier);
	
		// 입자정보를 SP2 에 복사
		cudaMemcpy(dev_SP2,dev_P2,sizeof(part2)*num_part2,cudaMemcpyDeviceToDevice);
		pthread_barrier_wait(&barrier);
	}





//-------------------------------------------------------------------------------------------------
// Eulerian: 질량 업데이트
//-------------------------------------------------------------------------------------------------

// if(count%(lap)==0){
	//if(dim==2) KERNEL_clc_volume_update<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3,count);
	if(dim==3) KERNEL_clc_volume_update3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3,count);
	//cudaDeviceSynchronize();
	KERNEL_clc_mass_update<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,count);
	cudaDeviceSynchronize();
	// }


	//-------------------------------------------------------------------------------------------------
	// 계산 준비: gradient correction, filter, reference density, p_type switch, penetration, density gradient
	//-------------------------------------------------------------------------------------------------

 	b.x=(num_part2-1)/t.x+1;



	// 미분보정필터 계산
	if((kgc_solve>0)||(delSPH_solve==Antuono))	gradient_correction(g_str,g_end,dev_SP1,dev_P3);

	// 표면장력 해석을 위한 color field 계산
	if(fs_solve){
		if(dim==2) KERNEL_clc_color_field2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
		if(dim==3) KERNEL_clc_color_field3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
		cudaDeviceSynchronize();
	}

	// filter, porosity (Due to the DEM Particles)
	if(dim==2) KERNEL_clc_prep2D_coupling<<<b,t>>>(g_str,g_end,dev_SP1, count);
	if(dim==3) KERNEL_clc_prep3D_coupling<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2,count, time);		//** 
	cudaDeviceSynchronize();

	// filter, reference density, p_type switch, penetration, normal gradient etc
	if(dim==2) KERNEL_clc_prep2D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
	if(dim==3) KERNEL_clc_prep3D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
	//if(dim==3) KERNEL_clc_prep3D_1<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);			//**
	cudaDeviceSynchronize();







	

	//-------------------------------------------------------------------------------------------------
	// 밀도 계산
	//-------------------------------------------------------------------------------------------------

	//KERNEL_switch_p_type_air<<<b,t>>>(1,dev_SP1,dev_P1, dev_SP2);
	//cudaDeviceSynchronize();

	if(rho_type==Mass_Sum){
		if(dim==2) KERNEL_clc_mass_sum_norm2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2);
		if(dim==3) KERNEL_clc_mass_sum_norm3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2);			//**
		cudaDeviceSynchronize();
	}
	else {
		if(dim==2) KERNEL_clc_continuity_norm2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		if(dim==3) KERNEL_clc_continuity_norm3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		//if(dim==3) KERNEL_clc_continuity_norm3D_1<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		//if((count>0)&&(freq_mass_sum>0)&&(count%freq_mass_sum==0)){
		//	if(dim==2) KERNEL_renormalization_norm2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		//	if(dim==3) KERNEL_renormalization_norm3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		//	cudaDeviceSynchronize();
		//}
	}


			


	//-------------------------------------------------------------------------------------------------
	// 압력 계산
	//-------------------------------------------------------------------------------------------------

	b.x=(num_part2-1)/t.x+1;
	KERNEL_EOS<<<b,t>>>(dev_SP1,dev_SP2);		//**
	cudaDeviceSynchronize();


	// 경계조건 (BOUNDARY CONDITION)
	if(noslip_bc==1){
		if(dim==2) KERNEL_boundary2D<<<b,t>>>(g_str,g_end,dev_SP1);
		if(dim==3) KERNEL_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1);
		cudaDeviceSynchronize();
	}

	if(dim==3) KERNEL_Neumann_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	//if(dim==3) KERNEL_Neumann_boundary3D_1<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	cudaDeviceSynchronize();


	// boundary_pressure
	//KERNEL_mirroring_adami<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);		// for boundary, moving
	//cudaDeviceSynchronize();


	//if (time>=0.0){

	//-------------------------------------------------------------------------------------------------
	// SPH 입자 상호작용 계산 (힘, 열전달, 확산)
	//-------------------------------------------------------------------------------------------------

	b.x=(num_part2-1)/t.x+1;
	//if(dim==2) KERNEL_interaction2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	if(dim==3) KERNEL_interaction3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	//if(dim==3) KERNEL_interaction3D_1<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);			// ** (유체 N-S + 유체 간 열전달 푸는 곳)
	cudaDeviceSynchronize();




	//}

	
	//if (time>=0.0){

	//-------------------------------------------------------------------------------------------------
	// DEM 입자 상호작용 계산 (충돌힘, 병진운동, 회전운동)
	//-------------------------------------------------------------------------------------------------

	b.x=(num_part2-1)/t.x+1;
	//if(dim==2) KERNEL_DEM_interaction2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);
	//if(dim==3) KERNEL_DEM_interaction3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P1,dev_P3,time);		// ** (DEM Dyanamics, heat transfer)
	cudaDeviceSynchronize();

	





	if (time>=0.0){



		//-------------------------------------------------------------------------------------------------
		// [SPH-DEM Coupling] DEM 입자 힘 계산 (압력구배힘, 항력 등)
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_coupling2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		if(dim==3) KERNEL_DEM_coupling3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);	// ** (DEM 입자가 받는 상호작용 힘, 열)
		cudaDeviceSynchronize();



		//-------------------------------------------------------------------------------------------------
		// [SPH-DEM Coupling] SPH 입자 힘 계산 (작용반작용)
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_SPH_coupling2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);
		if(dim==3) KERNEL_SPH_coupling3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);	// ** (SPH 입자가 받는 반작용 힘, 열)
		cudaDeviceSynchronize();

	
		
	}
	

	//-------------------------------------------------------------------------------------------------
	// 시간 적분 (Time Integration)
	//-------------------------------------------------------------------------------------------------

	b.x=(num_part2-1)/t.x+1;
	KERNEL_time_update_single<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P2,dev_P3,time,count);
	//KERNEL_time_update_single_1<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P2,dev_P3);

	//KERNEL_time_update_single_1(const Real tdt,part1*P1,part1*TP1,part2*P2,part2*TP2,part3*P3)
	cudaDeviceSynchronize();
	


	//-------------------------------------------------------------------------------------------------
	// DEM 경계 조건 (DEM Boundary Condition)
	//-------------------------------------------------------------------------------------------------

	// if (time<0.0){
	// 	KERNEL_treat_DEM_box_x1<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
	// }
	b.x=(num_part2-1)/t.x+1;
	//KERNEL_treat_DEM_cylinder_y<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
	//KERNEL_treat_DEM_cylinder_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
	//KERNEL_treat_DEM_cone_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3,time);
	//KERNEL_treat_DEM_cylinder_z1<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
	
	//KERNEL_treat_DEM_box_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3,time);

	//KERNEL_treat_DEM_cylinder_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
	//KERNEL_treat_DEM_box_z_simple<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3,time);

	cudaDeviceSynchronize();

	//if(time>1.0){
	b.x=(num_part2-1)/t.x+1;
	//KERNEL_treat_DEM_box_za<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3,time);
	cudaDeviceSynchronize();

	KERNEL_open_boundary_extrapolation3D_1<<<b,t>>>(time, g_str,g_end,dev_P1,dev_SP2,dev_P3,count,dt);
	//KERNEL_open_boundary_extrapolation3D_3<<<b,t>>>(time, g_str,g_end,dev_P1,dev_SP2,dev_P3,count,dt);
	cudaDeviceSynchronize();
	// KERNEL_open_boundary_extrapolation3D_2<<<b,t>>>(time, g_str,g_end,dev_P1,dev_SP2,dev_P3,count,dt);
	// cudaDeviceSynchronize();

	//KERNEL_time_update_buffer<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P3,space,Nsx,Nsz);
	//cudaDeviceSynchronize();

	// cudaMemcpy(file_P1,dev_P1,num_part2*sizeof(part1),cudaMemcpyDeviceToHost);
	// cudaMemcpy(file_P2,dev_SP2,num_part2*sizeof(part2),cudaMemcpyDeviceToHost);
	// cudaMemcpy(file_P3,dev_P3,num_part2*sizeof(part3),cudaMemcpyDeviceToHost);
	// save_plot_fluid_vtk_bin_fluid(file_P1,file_P3);		// fluid (SPH)

	// save_plot_fluid_vtk_bin_boundary(file_P1);	// boundary (SPH) -> 맨 첫 스텝에만 출력
	//-------------------------------------------------------------------------------------------------
	// CSF 검증
	//-------------------------------------------------------------------------------------------------

	b.x=(num_part2-1)/t.x+1;
	//	if(dim==2) KERNEL_clc_surface_detect2D<<<b,t>>>(g_str,g_end,dev_P1,dev_P3);
	//	if(dim==3) KERNEL_clc_surface_detect3D<<<b,t>>>(g_str,g_end,dev_P1,dev_P3);
	cudaDeviceSynchronize();

	//-------------------------------------------------------------------------------------------------
	// XSPH
	//-------------------------------------------------------------------------------------------------

	if(xsph_solve)
	{
		b.x=(num_part2-1)/t.x+1;
		if(dim==2) KERNEL_xsph2D<<<b,t>>>(1,g_str,g_end,dt,time,dev_SP1,dev_P1,dev_SP2);
		if(dim==3) KERNEL_xsph3D<<<b,t>>>(1,g_str,g_end,dt,time,dev_P1,dev_SP2);
		cudaDeviceSynchronize();
	}


	//-------------------------------------------------------------------------------------------------
	// Open Boudnary
	//-------------------------------------------------------------------------------------------------

	if(open_boundary>0)
	{
		

		if((open_boundary==1)&&(time>0.0)) {																						//(CAUTION)
			// if (dim==2) KERNEL_periodic_2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P1,dev_SP2);
			// if (dim==3) KERNEL_periodic_3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P1,dev_SP2);

			//KERNEL_open_boundary<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P3,space,Nsx,Nsz);
			cudaDeviceSynchronize();
		}
	}


	//-------------------------------------------------------------------------------------------------
	// 출력
	//-------------------------------------------------------------------------------------------------

	//if(((count%freq_output)==0) && (count>=0)){
	if(((count%freq_output)==0) ){
			printf("save plot...........................\n");
			cudaMemcpy(file_P1,dev_P1,num_part2*sizeof(part1),cudaMemcpyDeviceToHost);
			cudaMemcpy(file_P2,dev_SP2,num_part2*sizeof(part2),cudaMemcpyDeviceToHost);
			cudaMemcpy(file_P3,dev_P3,num_part2*sizeof(part3),cudaMemcpyDeviceToHost);
			save_plot_fluid_vtk_bin_fluid(file_P1,file_P2,file_P3);		// fluid (SPH)
			//save_plot_fluid_vtk_bin_air(file_P1,file_P2,file_P3);
			save_plot_fluid_vtk_bin_solid(file_P1,file_P3);		// solid (DEM)
		//	save_plot_fluid_vtk_bin_moving(file_P1,file_P3);
			//if (simulation_type==Two_Phase) save_plot_fluid_vtk_bin_fluid2(file_P1,file_P3);
			//if(count==0) save_plot_fluid_vtk_bin_boundary(file_P1);	// boundary (SPH) -> 맨 첫 스텝에만 출력
			save_plot_fluid_vtk_bin_boundary(file_P1);	// boundary (SPH) -> 맨 첫 스텝에만 출력
			//save_plot_fluid_vtk_bin_boundary(file_P1);

			// save_vtk_bin_single(file_P1,file_P2,file_P3);
			// if(surf_model==2) CSF_validation(file_P1,file_P3);
			// if(surf_model==1) IPF_validation(file_P1,file_P3);
			printf("time = %5.6f\n\n\n",time);
	 }

}


//-------------------------------------------------------------------------------------------------
// 모든 입자 계산을 수행하는 주함수
//-------------------------------------------------------------------------------------------------
void SOPHIA_multi(int_t*g_idx,int_t*p_idx,int_t*g_idx_in,int_t*p_idx_in,int_t*g_str,int_t*g_end,
									part1*dev_P1,part1*dev_SP1,part2*dev_P2,part2*dev_SP2,part3*dev_P3,
									int_t*p2p_af_in,int_t*p2p_idx_in,int_t*p2p_af,int_t*p2p_idx,
									void*dev_sort_storage,size_t*sort_storage_bytes,part1*file_P1,int tid)
{

	dim3 b,t;
	t.x=128;
	int s=sizeof(int)*(t.x+1);


	//-------------------------------------------------------------------------------------------------
	// PREDICTOR (Optional)
	//-------------------------------------------------------------------------------------------------
	if(time_type==Pre_Cor){
		b.x=(num_part2-1)/t.x+1;
		KERNEL_clc_predictor<<<b,t>>>(dt,time,dev_P1,dev_P2,dev_P3);
		cudaDeviceSynchronize();
	}


	//-------------------------------------------------------------------------------------------------
	// 주변입자 검색
	//-------------------------------------------------------------------------------------------------

		// g_str을 리셋
		cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells);

		// 입자의 i_type을 재계산
		b.x=(num_part2-1)/t.x+1;
		inner_outer_particle<<<b,t>>>(dev_P1,tid);
		cudaDeviceSynchronize();

		// 입자의 셀번호 계산
		b.x=(num_part2-1)/t.x+1;
		KERNEL_index_particle_to_cell<<<b,t>>>(g_idx_in,p_idx_in,dev_P1);
		cudaDeviceSynchronize();

		// 셀번호를 바탕으로 정렬
		cub::DeviceRadixSort::SortPairs(dev_sort_storage,*sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part2);
		cudaDeviceSynchronize();

		// 정렬한 입자를 재배치
		b.x=(num_part2-1)/t.x+1;
		KERNEL_reorder<<<b,t,s>>>(g_idx,p_idx,g_str,g_end,dev_P1,dev_P2,dev_SP1,dev_SP2);
		cudaDeviceSynchronize();

		// 일부 입자정보 리셋
		cudaMemset(dev_P3,0,sizeof(part3)*num_part2);

		// 입자정보를 P1 에 복사
		cudaMemcpy(dev_P1,dev_SP1,sizeof(part1)*num_part2,cudaMemcpyDeviceToDevice);
		pthread_barrier_wait(&barrier);


		//-------------------------------------------------------------------------------------------------
		// 계산 준비: gradient correction, filter, reference density, p_type switch, penetration, density gradient
		//-------------------------------------------------------------------------------------------------

	 	b.x=(num_part2-1)/t.x+1;

		// 미분보정필터 계산
		if((kgc_solve>0)||(delSPH_solve==Antuono))	gradient_correction(g_str,g_end,dev_SP1,dev_P3);


		// 표면장력 해석을 위한 color field 계산
		if(fs_solve){
			if(dim==2) KERNEL_clc_color_field2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
			if(dim==3) KERNEL_clc_color_field3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
			cudaDeviceSynchronize();
		}

		// filter, reference density, p_type switch, penetration, etc
		if(dim==2) KERNEL_clc_prep2D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
		if(dim==3) KERNEL_clc_prep3D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
		cudaDeviceSynchronize();

		// filter, porosity (Due to the DEM Particles)
		if(dim==2) KERNEL_clc_prep2D_coupling<<<b,t>>>(g_str,g_end,dev_SP1, count);
		if(dim==3) KERNEL_clc_prep3D_coupling<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2, count, time);
		cudaDeviceSynchronize();

		// 경계조건 (BOUNDARY CONDITION)
		if(noslip_bc==1){
			if(dim==2) KERNEL_boundary2D<<<b,t>>>(g_str,g_end,dev_SP1);
			if(dim==3) KERNEL_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1);
			cudaDeviceSynchronize();
		}


		//-------------------------------------------------------------------------------------------------
		// 밀도 계산
		//-------------------------------------------------------------------------------------------------

		if(rho_type==Mass_Sum){
			if(dim==2) KERNEL_clc_mass_sum_norm2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2);
			if(dim==3) KERNEL_clc_mass_sum_norm3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2);
			cudaDeviceSynchronize();
		}
		else {
			if(dim==2) KERNEL_clc_continuity_norm2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
			if(dim==3) KERNEL_clc_continuity_norm3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
			cudaDeviceSynchronize();

			if((count>0)&&(freq_mass_sum>0)&&(count%freq_mass_sum==0)){
				if(dim==2) KERNEL_renormalization_norm2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
				if(dim==3) KERNEL_renormalization_norm3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
				cudaDeviceSynchronize();
			}
		}


		//-------------------------------------------------------------------------------------------------
		// 압력 계산
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		KERNEL_EOS<<<b,t>>>(dev_SP1,dev_SP2);
		cudaDeviceSynchronize();


		//-------------------------------------------------------------------------------------------------
		// SPH 입자 상호작용 계산 (힘, 열전달, 확산) [영역 1]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		if(dim==2) KERNEL_interaction2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		if(dim==3) KERNEL_interaction3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// DEM 입자 상호작용 계산 (충돌힘, 병진운동, 회전운동) [영역 1]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_interaction2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);
		if(dim==3) KERNEL_DEM_interaction3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P1,dev_P3,time);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// DEM 입자 잠긴 부피 계산 [영역 1]
		//-------------------------------------------------------------------------------------------------
		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_SINKED_VOLUME2D<<<b,t>>>(1,g_str,g_end,dev_SP1);
		if(dim==3) KERNEL_DEM_SINKED_VOLUME3D<<<b,t>>>(1,g_str,g_end,dev_SP1);
		cudaDeviceSynchronize();

		
		//-------------------------------------------------------------------------------------------------
		// [SPH-DEM Coupling] DEM 입자 힘 계산 (압력구배힘, 항력 등) [영역 1]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_coupling2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		//if(dim==3) KERNEL_DEM_coupling3D_b<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_coupling2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		//if(dim==3) KERNEL_DEM_coupling3D_d<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// [SPH-DEM Coupling] SPH 입자 힘 계산 (작용반작용) [영역 1]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_SPH_coupling2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);
		if(dim==3) KERNEL_SPH_coupling3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);
		cudaDeviceSynchronize();


		//-------------------------------------------------------------------------------------------------
		// 시간 적분 (Time Integration) [영역 1]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		KERNEL_time_update<<<b,t>>>(1,dt,dev_SP1,dev_P1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// DEM 경계 조건 (DEM Boundary Condition) [영역 1]
  		//-------------------------------------------------------------------------------------------------

		// b.x=(num_part2-1)/t.x+1;
		// //KERNEL_treat_DEM_cylinder_y<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
		// KERNEL_treat_DEM_cylinder_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
		// KERNEL_treat_DEM_cylinder_z1<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
		// KERNEL_treat_DEM_box_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
		// KERNEL_treat_DEM_cone_z<<<b,t>>>(1,dev_SP1,dev_P1,dev_SP2,dev_P3);
		// cudaDeviceSynchronize();

		if(xsph_solve)
		{
			b.x=(num_part2-1)/t.x+1;
			if(dim==2) KERNEL_xsph2D<<<b,t>>>(1,g_str,g_end,dt,time,dev_SP1,dev_P1,dev_SP2);
			if(dim==3) KERNEL_xsph3D<<<b,t>>>(1,g_str,g_end,dt,time,dev_P1,dev_SP2);
			cudaDeviceSynchronize();
		}

		pthread_barrier_wait(&barrier);


		//-------------------------------------------------------------------------------------------------
		// GPU 간 데이터 전송을 위한 준비
		//-------------------------------------------------------------------------------------------------

		// 전송용 데이터 저장소를 초기화
		cudaMemset(send_rSP1[tid],0,sizeof(part1)*num_p2p);
		cudaMemset(send_lSP1[tid],0,sizeof(part1)*num_p2p);
		cudaMemset(send_rSP3[tid],0,sizeof(p2p_part3)*num_p2p);
		cudaMemset(send_lSP3[tid],0,sizeof(p2p_part3)*num_p2p);

		// 오른쪽 GPU로 전송할 데이터 준비
		if(tid!=ngpu-1){
			// 데이터 저장소 초기화
			cudaMemset(p2p_af_in,0,sizeof(int_t)*num_part2);
			cudaMemset(p2p_idx_in,0,sizeof(int_t)*num_part2);
			cudaMemset(send_P1[tid],0,sizeof(part1)*num_part2);
			cudaMemset(send_P3[tid],0,sizeof(p2p_part3)*num_part2);

			// 전송할 입자를 걸러내기 위한 표식 (p2p_af_in)
			b.x=(num_part2-1)/t.x+1;
			right_send_particle<<<b,t>>>(p2p_af_in,p2p_idx_in,dev_P1,dev_P3,send_P1[tid],send_P3[tid],tid);
			cudaDeviceSynchronize();

			// 표식을 기준으로 입자 정렬
			cub::DeviceRadixSort::SortPairs(dev_sort_storage,*sort_storage_bytes,p2p_af_in,p2p_af,p2p_idx_in,p2p_idx,num_part2);
			cudaDeviceSynchronize();

			// P2P 데이터 재배치
			b.x=(num_p2p-1)/t.x+1;
			reorder_data_p2p<<<b,t>>>(p2p_af,p2p_idx,send_P1[tid],send_rSP1[tid],send_P3[tid],send_rSP3[tid]);
			cudaDeviceSynchronize();
		}

		// 왼쪽으로 GPU 전송할 데이터 준비
		if(tid!=0){
			// 데이터 저장소 초기화
			cudaMemset(p2p_af_in,0,sizeof(int_t)*num_part2);
			cudaMemset(p2p_idx_in,0,sizeof(int_t)*num_part2);
			cudaMemset(send_P1[tid],0,sizeof(part1)*num_part2);
			cudaMemset(send_P3[tid],0,sizeof(p2p_part3)*num_part2);

			// 전송할 입자를 걸러내기 위한 표식 (p2p_af_in)
			b.x=(num_part2-1)/t.x+1;
			left_send_particle<<<b,t>>>(p2p_af_in,p2p_idx_in,dev_P1,dev_P3,send_P1[tid],send_P3[tid],tid);
			cudaDeviceSynchronize();

			// 표식을 기준으로 입자 정렬
			cub::DeviceRadixSort::SortPairs(dev_sort_storage,*sort_storage_bytes,p2p_af_in,p2p_af,p2p_idx_in,p2p_idx,num_part2);
			cudaDeviceSynchronize();

			// P2P 데이터 재배치
			b.x=(num_p2p-1)/t.x+1;
			reorder_data_p2p<<<b,t>>>(p2p_af,p2p_idx,send_P1[tid],send_lSP1[tid],send_P3[tid],send_lSP3[tid]);
			cudaDeviceSynchronize();
		}

		pthread_barrier_wait(&barrier);


		//-------------------------------------------------------------------------------------------------
		// GPU 간 데이터 전송 실행
		//-------------------------------------------------------------------------------------------------

		int half=(int)(num_p2p*0.5)-1;
		if(tid!=ngpu-1){
			cudaMemcpyPeerAsync(recv_P1[tid+1],tid+1,send_rSP1[tid],tid,sizeof(part1)*half,str2[tid]);
			cudaMemcpyPeerAsync(recv_P3[tid+1],tid+1,send_rSP3[tid],tid,sizeof(p2p_part3)*half,str2[tid]);
		}
		if(tid!=0){
			cudaMemcpyPeerAsync(&recv_P1[tid-1][half],tid-1,send_lSP1[tid],tid,sizeof(part1)*half,str2[tid]);
			cudaMemcpyPeerAsync(&recv_P3[tid-1][half],tid-1,send_lSP3[tid],tid,sizeof(p2p_part3)*half,str2[tid]);
		}


		//-------------------------------------------------------------------------------------------------
		// SPH 입자 힘 계산 [영역 0]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		if(dim==2) KERNEL_interaction2D<<<b,t,0,str1[tid]>>>(0,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		if(dim==3) KERNEL_interaction3D<<<b,t,0,str1[tid]>>>(0,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// DEM 입자 상호작용 계산 (충돌힘, 병진운동, 회전운동) [영역 0]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_interaction2D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_P3);
		if(dim==3) KERNEL_DEM_interaction3D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_P1,dev_P3,time);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// DEM 입자 잠긴 부피 계산 [영역 0]
		//-------------------------------------------------------------------------------------------------
		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_SINKED_VOLUME2D<<<b,t>>>(0,g_str,g_end,dev_SP1);
		if(dim==3) KERNEL_DEM_SINKED_VOLUME3D<<<b,t>>>(0,g_str,g_end,dev_SP1);
		cudaDeviceSynchronize();

		

		//-------------------------------------------------------------------------------------------------
		// [SPH-DEM Coupling] DEM 입자 힘 계산 (압력구배힘, 항력 등) [영역 0]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_DEM_coupling2D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		if(dim==3) KERNEL_DEM_coupling3D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();


		//-------------------------------------------------------------------------------------------------
		// [SPH-DEM Coupling] SPH 입자 힘 계산 (작용반작용) [영역 0]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		//if(dim==2) KERNEL_SPH_coupling2D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_P3);
		if(dim==3) KERNEL_SPH_coupling3D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_P3);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// 시간 적분 (Time Integration) [영역 0]
		//-------------------------------------------------------------------------------------------------

		b.x=(num_part2-1)/t.x+1;
		KERNEL_time_update<<<b,t>>>(0,dt,dev_SP1,dev_P1,dev_SP2,dev_P3);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// DEM 경계 조건 (DEM Boundary Condition) [영역 0]
  		//-------------------------------------------------------------------------------------------------

		  b.x=(num_part2-1)/t.x+1;
		  //KERNEL_treat_DEM_cylinder_y<<<b,t>>>(0,dev_SP1,dev_P1,dev_SP2,dev_P3);
		  KERNEL_treat_DEM_cylinder_z<<<b,t>>>(0,dev_SP1,dev_P1,dev_SP2,dev_P3);
		  //KERNEL_treat_DEM_box_z<<<b,t>>>(0,dev_SP1,dev_P1,dev_SP2,dev_P3);
		  cudaDeviceSynchronize();

		if(xsph_solve)
		{
			b.x=(num_part2-1)/t.x+1;
			if(dim==2) KERNEL_xsph2D<<<b,t>>>(0,g_str,g_end,dt,time,dev_SP1,dev_P1,dev_SP2);
			if(dim==3) KERNEL_xsph3D<<<b,t>>>(1,g_str,g_end,dt,time,dev_P1,dev_SP2);
			cudaDeviceSynchronize();
		}


		//-------------------------------------------------------------------------------------------------
		// p2p 데이터 전송
		//-------------------------------------------------------------------------------------------------

		// Recieve Data Copy to Each GPUs Particle Data
		b.x=(num_part2-1)/t.x+1;
		initial_particle<<<b,t>>>(dev_P1,tid);
		cudaDeviceSynchronize();

		pthread_barrier_wait(&barrier);

		// 각 GPU의 저장소(recv_P1, recv_P3)로 들어온 데이터를 각 입자들(dev_TSP1, dev_TSP3)로 복사
		b.x=(num_p2p-1)/t.x+1;
		p2p_copyData<<<b,t>>>(dev_P1,dev_P3,recv_P1[tid],recv_P3[tid],tid);
		cudaDeviceSynchronize();

		// 저장소의 정보를 0으로 리셋
		cudaMemset(recv_P3[tid],0,sizeof(p2p_part3)*num_p2p);

		pthread_barrier_wait(&barrier);


	//-------------------------------------------------------------------------------------------------
	// 출력
	//-------------------------------------------------------------------------------------------------
	if((count%freq_output)==0){
			printf("save plot...........................\n");
		 	//if(tid==0){
			// copy device to host
			cudaMemcpy(file_P1,dev_P1,num_part2*sizeof(part1),cudaMemcpyDeviceToHost);
			// ACSII
			//if(ngpu==1) save_plot_fluid_vtk2(file_P1);
			//else save_plot_fluid_vtk2_multi(file_P1,tid);

			//Binary
			// if(ngpu==1)	save_plot_fluid_vtk_bin(file_P1);			//save_plot_fluid_vtk_bin2(file_P1); //save_plot_fluid_vtk_bin_test(file_P1);
			// else save_plot_fluid_vtk_bin3_test(file_P1,tid);	//save_plot_fluid_vtk_bin3(file_P1,tid);
			//else save_plot_fluid_vtk_bin3_test(file_P1,tid);
		 //}
		 	printf("time = %5.6f\n\n\n",time);
	}
	pthread_barrier_wait(&barrier);
}


//-------------------------------------------------------------------------------------------------
// SOPHIA 메인 코드
//-------------------------------------------------------------------------------------------------
void*WCSPH_Calc(void*arg){

	// 함수의 인자를 받아서 tid 에 저장 (tid = gpu 번호)
	int*idPtr,tid;
	idPtr=(int*)arg;
	tid=*idPtr;

	// timestep control 을 위한 변수 설정
	Real dt_CFL,V_MAX,K_stiff,eta;
	Real h0=HP1[0].h;

	dt_CFL=V_MAX=K_stiff=eta=0.0;
	num_cells=clc_num_cells();

	count=floor(time/dt+0.5);


	//-------------------------------------------------------------------------------------------------
	// Device 입자 생성
	//-------------------------------------------------------------------------------------------------

	// 계산할 GPU 정의: tid=GPU number
	cudaSetDevice(tid);

	// GPU 내에 분기 생성 : stream is for run the kernel and memcpy peer to peer at the same time.
	cudaStreamCreate(&str1[tid]);
	cudaStreamCreate(&str2[tid]);

	// 출력할 변수 선언 및 메모리 할당
	part1*file_P1;
	file_P1=(part1*)malloc(sizeof(part1)*num_part2);
	memset(file_P1,0,sizeof(part1)*num_part2);

	part2*file_P2;
	part3*file_P3;

	file_P2=(part2*)malloc(sizeof(part2)*num_part2);
	memset(file_P2,0,sizeof(part2)*num_part2);

	file_P3=(part3*)malloc(sizeof(part3)*num_part2);
	memset(file_P3,0,sizeof(part3)*num_part2);

	//-------------------------------------------------------------------------------------------------
	// Device/GPU 변수 선언 및 메모리 할당
	//-------------------------------------------------------------------------------------------------

	// NNPS 관련 변수
	int_t*g_idx,*p_idx,*g_idx_in,*p_idx_in,*g_str,*g_end;

	// 주요 입자 변수
	part1*dev_P1,*dev_SP1;
	part2*dev_P2,*dev_SP2;
	part3*dev_SP3;

	// P2P 데이터 변수 선언
	int*p2p_af_in,*p2p_idx_in,*p2p_af,*p2p_idx;

	// NNPS 입자 메모리 할당
	cudaMalloc((void**)&g_idx,sizeof(int_t)*num_part2);
	cudaMalloc((void**)&p_idx,sizeof(int_t)*num_part2);
	cudaMalloc((void**)&g_idx_in,sizeof(int_t)*num_part2);
	cudaMalloc((void**)&p_idx_in,sizeof(int_t)*num_part2);
	cudaMalloc((void**)&g_str,sizeof(int_t)*num_cells);
	cudaMalloc((void**)&g_end,sizeof(int_t)*num_cells);

	// Device 입자 데이터 메모리 할당
	cudaMalloc((void**)&dev_P1,sizeof(part1)*num_part2);
	cudaMalloc((void**)&dev_SP1,sizeof(part1)*num_part2);
	cudaMalloc((void**)&dev_P2,sizeof(part2)*num_part2);
	cudaMalloc((void**)&dev_SP2,sizeof(part2)*num_part2);
	cudaMalloc((void**)&dev_SP3,sizeof(part3)*num_part2);

	// NNPS 메모리 초기화
	cudaMemset(g_idx_in,0,sizeof(int_t)*num_part2);
	cudaMemset(p_idx_in,0,sizeof(int_t)*num_part2);
	cudaMemset(g_idx,0,sizeof(int_t)*num_part2);
	cudaMemset(p_idx,0,sizeof(int_t)*num_part2);
	cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells);
	cudaMemset(g_end,0,sizeof(int_t)*num_cells);

	// Device 입자 메모리 초기화
	cudaMemset(dev_P1,0,sizeof(part1)*num_part2);
	cudaMemset(dev_SP1,0,sizeof(part1)*num_part2);
	cudaMemset(dev_P2,0,sizeof(part2)*num_part2);
	cudaMemset(dev_SP2,0,sizeof(part2)*num_part2);
	cudaMemset(dev_SP3,0,sizeof(part3)*num_part2);

	// P2P 데이터 메모리 할당
	if(ngpu>1){
		// P2P Data Sorting Variable
		cudaMalloc((void**)&p2p_af_in,sizeof(int_t)*num_part2);
		cudaMalloc((void**)&p2p_idx_in,sizeof(int_t)*num_part2);
		cudaMalloc((void**)&p2p_af,sizeof(int_t)*num_part2);
		cudaMalloc((void**)&p2p_idx,sizeof(int_t)*num_part2);

		// P2P Data Send Variable (Unsorting)
		cudaMalloc((void**)&send_P1[tid],sizeof(part1)*num_part2);
		cudaMalloc((void**)&send_P3[tid],sizeof(p2p_part3)*num_part2);

		// P2P Data Right, Left Send Variable (Sorting)
		cudaMalloc((void**)&send_rSP1[tid],sizeof(part1)*num_p2p);
		cudaMalloc((void**)&send_lSP1[tid],sizeof(part1)*num_p2p);
		cudaMalloc((void**)&send_rSP3[tid],sizeof(p2p_part3)*num_p2p);
		cudaMalloc((void**)&send_lSP3[tid],sizeof(p2p_part3)*num_p2p);

		// P2P Data Recieve Variable
		cudaMalloc((void**)&recv_P1[tid],sizeof(part1)*num_p2p);
		cudaMalloc((void**)&recv_P3[tid],sizeof(p2p_part3)*num_p2p);

		cudaMemset(p2p_af,0,sizeof(int_t)*num_part2);
		cudaMemset(p2p_idx,0,sizeof(int_t)*num_part2);
		cudaMemset(p2p_af_in,0,sizeof(int_t)*num_part2);
		cudaMemset(p2p_idx_in,0,sizeof(int_t)*num_part2);
		cudaMemset(send_P1[tid],0,sizeof(part1)*num_part2);
		cudaMemset(send_P3[tid],0,sizeof(p2p_part3)*num_part2);
		cudaMemset(send_rSP1[tid],0,sizeof(part1)*num_p2p);
		cudaMemset(send_lSP1[tid],0,sizeof(part1)*num_p2p);
		cudaMemset(send_rSP3[tid],0,sizeof(p2p_part3)*num_p2p);
		cudaMemset(send_lSP3[tid],0,sizeof(p2p_part3)*num_p2p);
		cudaMemset(recv_P1[tid],0,sizeof(part1)*num_p2p);
		cudaMemset(recv_P3[tid],0,sizeof(p2p_part3)*num_p2p);
	}


	//-------------------------------------------------------------------------------------------------
	// Device/GPU로 데이터 복사
	//-------------------------------------------------------------------------------------------------

	// Sovler 전역변수 Device로 복사
	cudaMemcpyToSymbol(k_vii,vii,sizeof(int_t)*vii_size);
	cudaMemcpyToSymbol(k_vif,vif,sizeof(Real)*vif_size);

	// 물성 Table 데이터 Device로 복사
	cudaMemcpyToSymbol(k_Tab_T,host_Tab_T,sizeof(Real)*table_size);
	cudaMemcpyToSymbol(k_Tab_h,host_Tab_h,sizeof(Real)*table_size);
	cudaMemcpyToSymbol(k_Tab_k,host_Tab_k,sizeof(Real)*table_size);
	cudaMemcpyToSymbol(k_Tab_vis,host_Tab_vis,sizeof(Real)*table_size);

	cudaMemcpyToSymbol(k_table_index,host_table_index,sizeof(int)*10);
	cudaMemcpyToSymbol(k_table_size,host_table_size,sizeof(int)*10);

	// Host 입자정보(HP1)를 분할하여(DHP1) Device로 복사(dev_P1)
	DHP1[tid]=(part1*)malloc(num_part2*sizeof(part1));
	memset(DHP1[tid],0,sizeof(part1)*num_part2);

	// 모든 입자를 더미로 초기화
	for(int i=0;i<num_part2;i++) DHP1[tid][i].i_type=3;

	// 입자 분할(HP1->DHP1) 및 입자 Type 설정(i_type, buffer_type)
	if(ngpu>1){
		c_initial_inner_outer_particle(HP1,DHP1[tid],tid);
		cudaMemcpy(dev_P1,DHP1[tid],num_part2*sizeof(part1),cudaMemcpyHostToDevice);
	}else{
		c_initial_inner_outer_particle_single(HP1,DHP1[tid],tid);											// (CAUTION)
		cudaMemcpy(dev_P1,DHP1[tid],num_part2*sizeof(part1),cudaMemcpyHostToDevice);	// single gpu 이면 그냥 HP1을 device에 복사
	}
	pthread_barrier_wait(&barrier);

	if(tid==0){
		printf("\n-----------------------------------------------------------\n");
		printf("GPU Domain Division Success\n");
		printf("-----------------------------------------------------------\n\n");
	}


	//-------------------------------------------------------------------------------------------------
	// 화면 출력용 기타 변수들 정의 및 메모리 할당 (최대속도, 최대힘 등)
	//-------------------------------------------------------------------------------------------------

	// host
	Real *max_umag0,*max_rho0,*max_ftotal0;
	max_umag0=(Real*)malloc(sizeof(Real));
	max_rho0=(Real*)malloc(sizeof(Real));
	max_ftotal0=(Real*)malloc(sizeof(Real));
	max_umag0[0]=max_ftotal0[0]=max_rho0[0]=0.0;

	// device
	Real*max_rho,*max_umag,*max_ft,*d_max_umag0,*d_max_rho0,*d_max_ftotal0;
	cudaMalloc((void**)&max_rho,sizeof(Real)*num_part2);
	cudaMalloc((void**)&max_umag,sizeof(Real)*num_part2);
	cudaMalloc((void**)&max_ft,sizeof(Real)*num_part2);
	cudaMalloc((void**)&d_max_umag0,sizeof(Real));
	cudaMalloc((void**)&d_max_rho0,sizeof(Real));
	cudaMalloc((void**)&d_max_ftotal0,sizeof(Real));
	cudaMemset(max_umag,0,sizeof(Real)*num_part2);
	cudaMemset(max_rho,0,sizeof(Real)*num_part2);
	cudaMemset(max_ft,0,sizeof(Real)*num_part2);
	cudaMemset(d_max_umag0,0,sizeof(Real));
	cudaMemset(d_max_rho0,0,sizeof(Real));
	cudaMemset(d_max_ftotal0,0,sizeof(Real));


	//-------------------------------------------------------------------------------------------------
	// 정렬(Sorting)을 위한 CUB 라이브러리 변수 준비
	//-------------------------------------------------------------------------------------------------

	// Sorting & Max variable to use CUB Library
	void*dev_sort_storage=NULL;
	void*dev_max_storage=NULL;
	size_t sort_storage_bytes=0;
	size_t max_storage_bytes=0;

	// Determine Sorting & Maximum Value Setting for Total Particle Data
	cub::DeviceRadixSort::SortPairs(dev_sort_storage,sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part2);
	cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part2);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&dev_sort_storage,sort_storage_bytes);
	cudaMalloc((void**)&dev_max_storage,max_storage_bytes);
	pthread_barrier_wait(&barrier);


	//-------------------------------------------------------------------------------------------------
	// 주변입자 검색(NNPS)을 위한 변수 준비: g_str, g_end, SP1, SP2, SP3
	//-------------------------------------------------------------------------------------------------
	dim3 b,t;
	t.x=256;

	// gpu간 데이터 교환용 변수를 더미로 설정(i_type=3)
	if(ngpu>1){
		b.x=(num_p2p-1)/t.x+1;
		init_Recv<<<b,t>>>(recv_P1[tid]);
		cudaDeviceSynchronize();
	}

	pthread_barrier_wait(&barrier);


	//-------------------------------------------------------------------------------------------------
	// 코드 메인
	//-------------------------------------------------------------------------------------------------

	// 초기상태 및 설정 출력
	if(tid==0){
		printf("-----------------------------------------------------------\n");
		printf("Input Summary: \n");
		printf("-----------------------------------------------------------\n");
		printf("	Total number of particles=%d\n",num_part);
		printf("	Device number of particles=%d\n",num_part2);
		printf("	P2P number of particles=%d\n",num_p2p);
		printf("	NI=%d,	NJ=%d,	NK=%d\n",NI,NJ,NK);
		printf("-----------------------------------------------------------\n\n");
		// Input Check
		printf("-----------------------------------------------------------\n");
		printf("Input Check: \n");
		printf("-----------------------------------------------------------\n");
		// check Domain Status
		printf("x min, max : %f %f\n",x_min,x_max);
		printf("y min, max : %f %f\n",y_min,y_max);
		printf("z min, max : %f %f\n",z_min,z_max);
		printf("Cell Size(dcell) %f\n",dcell);
		printf("Number of Cells Per a GPU in x-direction(calc_area) %d\n",calc_area);
		printf("-----------------------------------------------------------\n\n");
		// print out status
		printf("\n");
		printf("-----------------------------\n");
		printf("Start Simultion!!\n");
		printf("-----------------------------\n");
		printf("\n");
	}
	pthread_barrier_wait(&barrier);

	//-------------------------------------------------------------------------------------------------
	// 코드 메인
	//-------------------------------------------------------------------------------------------------

	t.x=128;
	b.x=(num_part2-1)/t.x+1;
	KERNEL_clc_TemptoEnthalpy<<<b,t>>>(dev_P1,dev_P2);
	cudaDeviceSynchronize();

	while(1){

		if(ngpu==1){
			SOPHIA_single(g_idx,p_idx,g_idx_in,p_idx_in,g_str,g_end,dev_P1,dev_SP1,dev_P2,dev_SP2,dev_SP3,
									p2p_af_in,p2p_idx_in,p2p_af,p2p_idx,dev_sort_storage,&sort_storage_bytes,file_P1,file_P2,file_P3,tid);
		}
		else{
			SOPHIA_multi(g_idx,p_idx,g_idx_in,p_idx_in,g_str,g_end,dev_P1,dev_SP1,dev_P2,dev_SP2,dev_SP3,
									p2p_af_in,p2p_idx_in,p2p_af,p2p_idx,dev_sort_storage,&sort_storage_bytes,file_P1,tid);
		}


		//-------------------------------------------------------------------------------------------------
		// Time-step Control
		//-------------------------------------------------------------------------------------------------
		if(tid==0){
			time+=dt;
			count++;

			//timestep is updated every 10 steps ------------ estimate new timestep (Goswami & Pajarola(2011))
			if((count%(freq_output/10))==0){
				t.x=128;
				b.x=(num_part2-1)/t.x+1;
				kernel_copy_max<<<b,t>>>(dev_P1,dev_SP3,max_rho,max_ft,max_umag);
				cudaDeviceSynchronize();

				// Find Max Velocity & Force using CUB - TID=0
				cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part2);
				cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_rho,d_max_rho0,num_part2);
				cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_ft,d_max_ftotal0,num_part2);
				cudaDeviceSynchronize();
				cudaMemcpy(max_umag0,d_max_umag0,sizeof(Real),cudaMemcpyDeviceToHost);
				cudaMemcpy(max_rho0,d_max_rho0,sizeof(Real),cudaMemcpyDeviceToHost);
				cudaMemcpy(max_ftotal0,d_max_ftotal0,sizeof(Real),cudaMemcpyDeviceToHost);

				printf("%d\t rho_max=%5.2f\tu_max=%5.2f\tftotal_max=%5.2f\n\n",count,max_rho0[0],max_umag0[0],max_ftotal0[0]);
			}
		}

		pthread_barrier_wait(&barrier);
		if(time>=time_end) break;

	}


	//-------------------------------------------------------------------------------------------------
	// ##. Save Restart File
	//-------------------------------------------------------------------------------------------------

	if(ngpu==1) {

		cudaMemcpy(file_P1,dev_SP1,num_part2*sizeof(part1),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P2,dev_SP2,num_part2*sizeof(part2),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P3,dev_SP3,num_part2*sizeof(part3),cudaMemcpyDeviceToHost);

		save_restart(file_P1,file_P2,file_P3);

		free(file_P2);
		free(file_P3);
	}

	//-------------------------------------------------------------------------------------------------
	// ##. Memory Free
	//-------------------------------------------------------------------------------------------------
	free(file_P1);
	free(max_umag0);
	free(max_rho0);
	free(max_ftotal0);
	cudaFree(g_idx);
	cudaFree(p_idx);
	cudaFree(g_idx_in);
	cudaFree(p_idx_in);
	cudaFree(g_str);
	cudaFree(g_end);
	cudaFree(dev_P1);
	cudaFree(dev_SP1);
	cudaFree(dev_P2);
	cudaFree(dev_SP2);
	cudaFree(dev_SP3);
	cudaFree(max_umag);
	cudaFree(max_rho);
	cudaFree(max_ft);
	cudaFree(d_max_umag0);
	cudaFree(d_max_rho0);
	cudaFree(d_max_ftotal0);
	cudaFree(dev_sort_storage);
	cudaFree(dev_max_storage);
	if(ngpu>1){
		cudaFree(p2p_af);
		cudaFree(p2p_idx);
		cudaFree(p2p_af_in);
		cudaFree(p2p_idx_in);
		cudaFree(send_P1);
		cudaFree(send_P3);
		cudaFree(send_rSP1);
		cudaFree(send_lSP1);
		cudaFree(send_rSP3);
		cudaFree(send_lSP3);
		cudaFree(recv_P1);
		cudaFree(recv_P3);
	}
	cudaStreamDestroy(str1[tid]);
	cudaStreamDestroy(str2[tid]);
	pthread_barrier_wait(&barrier);

	return 0;
}
