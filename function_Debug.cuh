#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
void Coupled_Decoupled_check_sph(part1 *dev_SP1, part1 *dev_SP1_sph, part2 *dev_SP2, part2 *dev_SP2_sph, part3 *dev_P3, part3 *dev_P3_sph, int_t *g_idx_in, int_t *g_idx_in_sph)
{
	// dev_SP1의 p_type < 1001인 입자와 dev_SP1_sph의 값 비교 디버깅
	part1 *host_SP1 = (part1*)malloc(sizeof(part1)*num_part2);
	part2 *host_SP2 = (part2*)malloc(sizeof(part2)*num_part2);
	part3 *host_P3 = (part3*)malloc(sizeof(part3)*num_part2);
	part1 *host_SP1_sph = (part1*)malloc(sizeof(part1)*num_part2_sph);
	part2 *host_SP2_sph = (part2*)malloc(sizeof(part2)*num_part2_sph);
	part3 *host_P3_sph = (part3*)malloc(sizeof(part3)*num_part2_sph);
	int_t *host_g_idx_in = (int_t*)malloc(sizeof(int_t)*num_part2);
	int_t *host_g_idx_in_sph = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
	CUDA_CHECK(cudaMemcpy(host_SP1, dev_SP1, sizeof(part1)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_SP2, dev_SP2, sizeof(part2)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_P3, dev_P3, sizeof(part3)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_SP1_sph, dev_SP1_sph, sizeof(part1)*num_part2_sph, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_SP2_sph, dev_SP2_sph, sizeof(part2)*num_part2_sph, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_P3_sph, dev_P3_sph, sizeof(part3)*num_part2_sph, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_g_idx_in, g_idx_in, sizeof(int_t)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_g_idx_in_sph, g_idx_in_sph, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());
	int mismatch_count = 0;
	int coupled_idx = 0;
	int decoupled_idx = 0;

	for(int i=0; i<num_part2; i++) {
		if(host_SP1[i].p_type <= 1000) {
			int found = 0;
			if(host_g_idx_in[coupled_idx] != host_g_idx_in_sph[decoupled_idx]){
				printf("[DEBUG] SPH mismatched g_idx_in value coupled_idx=%d, decoupled_idx=%d, g_idx_in_coupled=%d, g_idx_in_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_g_idx_in[coupled_idx], host_g_idx_in_sph[decoupled_idx]);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].i_type != host_SP1_sph[decoupled_idx].i_type) {
				printf("[DEBUG] SPH mismatched i_type value coupled_idx=%d, decoupled_idx=%d, i_type_coupled=%d, i_type_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].i_type, host_SP1_sph[decoupled_idx].i_type);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].buffer_type != host_SP1_sph[decoupled_idx].buffer_type) {
				printf("[DEBUG] SPH mismatched buffer_type value coupled_idx=%d, decoupled_idx=%d, buffer_type_coupled=%d, buffer_type_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].buffer_type, host_SP1_sph[decoupled_idx].buffer_type);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].p_type != host_SP1_sph[decoupled_idx].p_type) {
				printf("[DEBUG] SPH mismatched p_type value coupled_idx=%d, decoupled_idx=%d, p_type_coupled=%d, p_type_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].p_type, host_SP1_sph[decoupled_idx].p_type);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].dem_idx != host_SP1_sph[decoupled_idx].dem_idx) {
				printf("[DEBUG] SPH mismatched dem_idx value coupled_idx=%d, decoupled_idx=%d, dem_idx_coupled=%d, dem_idx_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].dem_idx, host_SP1_sph[decoupled_idx].dem_idx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].ct_boundary != host_SP1_sph[decoupled_idx].ct_boundary) {
				printf("[DEBUG] SPH mismatched ct_boundary value coupled_idx=%d, decoupled_idx=%d, ct_boundary_coupled=%d, ct_boundary_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].ct_boundary, host_SP1_sph[decoupled_idx].ct_boundary);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].x != host_SP1_sph[decoupled_idx].x){
				printf("[DEBUG] SPH mismatched x value coupled_idx=%d, decoupled_idx=%d, x_coupled=%f, x_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].x, host_SP1_sph[decoupled_idx].x);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].y != host_SP1_sph[decoupled_idx].y){
				printf("[DEBUG] SPH mismatched y value coupled_idx=%d, decoupled_idx=%d, y_coupled=%f, y_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].y, host_SP1_sph[decoupled_idx].y);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].z != host_SP1_sph[decoupled_idx].z){
				printf("[DEBUG] SPH mismatched z value coupled_idx=%d, decoupled_idx=%d, z_coupled=%f, z_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].z, host_SP1_sph[decoupled_idx].z);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].x_star != host_SP1_sph[decoupled_idx].x_star){
				printf("[DEBUG] SPH mismatched x_star value coupled_idx=%d, decoupled_idx=%d, x_star_coupled=%f, x_star_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].x_star, host_SP1_sph[decoupled_idx].x_star);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].y_star != host_SP1_sph[decoupled_idx].y_star){
				printf("[DEBUG] SPH mismatched y_star value coupled_idx=%d, decoupled_idx=%d, y_star_coupled=%f, y_star_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].y_star, host_SP1_sph[decoupled_idx].y_star);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].z_star != host_SP1_sph[decoupled_idx].z_star){
				printf("[DEBUG] SPH mismatched z_star value coupled_idx=%d, decoupled_idx=%d, z_star_coupled=%f, z_star_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].z_star, host_SP1_sph[decoupled_idx].z_star);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].ux != host_SP1_sph[decoupled_idx].ux){
				printf("[DEBUG] SPH mismatched ux value coupled_idx=%d, decoupled_idx=%d, ux_coupled=%f, ux_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].ux, host_SP1_sph[decoupled_idx].ux);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].uy != host_SP1_sph[decoupled_idx].uy){
				printf("[DEBUG] SPH mismatched uy value coupled_idx=%d, decoupled_idx=%d, uy_coupled=%f, uy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].uy, host_SP1_sph[decoupled_idx].uy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].uz != host_SP1_sph[decoupled_idx].uz){
				printf("[DEBUG] SPH mismatched uz value coupled_idx=%d, decoupled_idx=%d, uz_coupled=%f, uz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].uz, host_SP1_sph[decoupled_idx].uz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].wx != host_SP1_sph[decoupled_idx].wx){
				printf("[DEBUG] SPH mismatched wx value coupled_idx=%d, decoupled_idx=%d, wx_coupled=%f, wx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].wx, host_SP1_sph[decoupled_idx].wx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].wy != host_SP1_sph[decoupled_idx].wy){
				printf("[DEBUG] SPH mismatched wy value coupled_idx=%d, decoupled_idx=%d, wy_coupled=%f, wy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].wy, host_SP1_sph[decoupled_idx].wy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].wz != host_SP1_sph[decoupled_idx].wz){
				printf("[DEBUG] SPH mismatched wz value coupled_idx=%d, decoupled_idx=%d, wz_coupled=%f, wz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].wz, host_SP1_sph[decoupled_idx].wz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].m != host_SP1_sph[decoupled_idx].m){
				printf("[DEBUG] SPH mismatched m value coupled_idx=%d, decoupled_idx=%d, m_coupled=%f, m_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].m, host_SP1_sph[decoupled_idx].m);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].ri != host_SP1_sph[decoupled_idx].ri){
				printf("[DEBUG] SPH mismatched ri value coupled_idx=%d, decoupled_idx=%d, ri_coupled=%f, ri_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].ri, host_SP1_sph[decoupled_idx].ri);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].rad != host_SP1_sph[decoupled_idx].rad){
				printf("[DEBUG] SPH mismatched rad value coupled_idx=%d, decoupled_idx=%d, rad_coupled=%f, rad_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].rad, host_SP1_sph[decoupled_idx].rad);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].h != host_SP1_sph[decoupled_idx].h){
				printf("[DEBUG] SPH mismatched h value coupled_idx=%d, decoupled_idx=%d, h_coupled=%f, h_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].h, host_SP1_sph[decoupled_idx].h);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp != host_SP1_sph[decoupled_idx].temp){
				printf("[DEBUG] SPH mismatched temp value coupled_idx=%d, decoupled_idx=%d, temp_coupled=%f, temp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp, host_SP1_sph[decoupled_idx].temp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp1 != host_SP1_sph[decoupled_idx].temp1){
				printf("[DEBUG] SPH mismatched temp1 value coupled_idx=%d, decoupled_idx=%d, temp1_coupled=%f, temp1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp1, host_SP1_sph[decoupled_idx].temp1);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp2 != host_SP1_sph[decoupled_idx].temp2){
				printf("[DEBUG] SPH mismatched temp2 value coupled_idx=%d, decoupled_idx=%d, temp2_coupled=%f, temp2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp2, host_SP1_sph[decoupled_idx].temp2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp3 != host_SP1_sph[decoupled_idx].temp3){
				printf("[DEBUG] SPH mismatched temp3 value coupled_idx=%d, decoupled_idx=%d, temp3_coupled=%f, temp3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp3, host_SP1_sph[decoupled_idx].temp3);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres != host_SP1_sph[decoupled_idx].pres){
				printf("[DEBUG] SPH mismatched pres value coupled_idx=%d, decoupled_idx=%d, pres_coupled=%f, pres_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres, host_SP1_sph[decoupled_idx].pres);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].rho != host_SP1_sph[decoupled_idx].rho){
				printf("[DEBUG] SPH mismatched rho value coupled_idx=%d, decoupled_idx=%d, rho_coupled=%f, rho_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].rho, host_SP1_sph[decoupled_idx].rho);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].flt_s != host_SP1_sph[decoupled_idx].flt_s){
				printf("[DEBUG] SPH mismatched flt_s value coupled_idx=%d, decoupled_idx=%d, flt_s_coupled=%f, flt_s_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].flt_s, host_SP1_sph[decoupled_idx].flt_s);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].flt_sd != host_SP1_sph[decoupled_idx].flt_sd){
				printf("[DEBUG] SPH mismatched flt_sd value coupled_idx=%d, decoupled_idx=%d, flt_sd_coupled=%f, flt_sd_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].flt_sd, host_SP1_sph[decoupled_idx].flt_sd);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].flt_sd_2 != host_SP1_sph[decoupled_idx].flt_sd_2){
				printf("[DEBUG] SPH mismatched flt_sd_2 value coupled_idx=%d, decoupled_idx=%d, flt_sd_2_coupled=%f, flt_sd_2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].flt_sd_2, host_SP1_sph[decoupled_idx].flt_sd_2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].w_dx != host_SP1_sph[decoupled_idx].w_dx){
				printf("[DEBUG] SPH mismatched w_dx value coupled_idx=%d, decoupled_idx=%d, w_dx_coupled=%f, w_dx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].w_dx, host_SP1_sph[decoupled_idx].w_dx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].enthalpy != host_SP1_sph[decoupled_idx].enthalpy){
				printf("[DEBUG] SPH mismatched enthalpy value coupled_idx=%d, decoupled_idx=%d, enthalpy_coupled=%f, enthalpy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].enthalpy, host_SP1_sph[decoupled_idx].enthalpy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].concn != host_SP1_sph[decoupled_idx].concn){
				printf("[DEBUG] SPH mismatched concn value coupled_idx=%d, decoupled_idx=%d, concn_coupled=%f, concn_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].concn, host_SP1_sph[decoupled_idx].concn);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].grad_rhox != host_SP1_sph[decoupled_idx].grad_rhox){
				printf("[DEBUG] SPH mismatched grad_rhox value coupled_idx=%d, decoupled_idx=%d, grad_rhox_coupled=%f, grad_rhox_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].grad_rhox, host_SP1_sph[decoupled_idx].grad_rhox);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].grad_rhoy != host_SP1_sph[decoupled_idx].grad_rhoy){
				printf("[DEBUG] SPH mismatched grad_rhoy value coupled_idx=%d, decoupled_idx=%d, grad_rhoy_coupled=%f, grad_rhoy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].grad_rhoy, host_SP1_sph[decoupled_idx].grad_rhoy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].grad_rhoz != host_SP1_sph[decoupled_idx].grad_rhoz){
				printf("[DEBUG] SPH mismatched grad_rhoz value coupled_idx=%d, decoupled_idx=%d, grad_rhoz_coupled=%f, grad_rhoz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].grad_rhoz, host_SP1_sph[decoupled_idx].grad_rhoz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pgf_x != host_SP1_sph[decoupled_idx].pgf_x){
				printf("[DEBUG] SPH mismatched pgf_x value coupled_idx=%d, decoupled_idx=%d, pgf_x_coupled=%f, pgf_x_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pgf_x, host_SP1_sph[decoupled_idx].pgf_x);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pgf_y != host_SP1_sph[decoupled_idx].pgf_y){
				printf("[DEBUG] SPH mismatched pgf_y value coupled_idx=%d, decoupled_idx=%d, pgf_y_coupled=%f, pgf_y_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pgf_y, host_SP1_sph[decoupled_idx].pgf_y);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pgf_z != host_SP1_sph[decoupled_idx].pgf_z){
				printf("[DEBUG] SPH mismatched pgf_z value coupled_idx=%d, decoupled_idx=%d, pgf_z_coupled=%f, pgf_z_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pgf_z, host_SP1_sph[decoupled_idx].pgf_z);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdx_b != host_SP1_sph[decoupled_idx].Fdx_b){
				printf("[DEBUG] SPH mismatched Fdx_b value coupled_idx=%d, decoupled_idx=%d, Fdx_b_coupled=%f, Fdx_b_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdx_b, host_SP1_sph[decoupled_idx].Fdx_b);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdy_b != host_SP1_sph[decoupled_idx].Fdy_b){
				printf("[DEBUG] SPH mismatched Fdy_b value coupled_idx=%d, decoupled_idx=%d, Fdy_b_coupled=%f, Fdy_b_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdy_b, host_SP1_sph[decoupled_idx].Fdy_b);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdz_b != host_SP1_sph[decoupled_idx].Fdz_b){
				printf("[DEBUG] SPH mismatched Fdz_b value coupled_idx=%d, decoupled_idx=%d, Fdz_b_coupled=%f, Fdz_b_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdz_b, host_SP1_sph[decoupled_idx].Fdz_b);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdx_df != host_SP1_sph[decoupled_idx].Fdx_df){
				printf("[DEBUG] SPH mismatched Fdx_df value coupled_idx=%d, decoupled_idx=%d, Fdx_df_coupled=%f, Fdx_df_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdx_df, host_SP1_sph[decoupled_idx].Fdx_df);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdy_df != host_SP1_sph[decoupled_idx].Fdy_df){
				printf("[DEBUG] SPH mismatched Fdy_df value coupled_idx=%d, decoupled_idx=%d, Fdy_df_coupled=%f, Fdy_df_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdy_df, host_SP1_sph[decoupled_idx].Fdy_df);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdz_df != host_SP1_sph[decoupled_idx].Fdz_df){
				printf("[DEBUG] SPH mismatched Fdz_df value coupled_idx=%d, decoupled_idx=%d, Fdz_df_coupled=%f, Fdz_df_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdz_df, host_SP1_sph[decoupled_idx].Fdz_df);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdx_da != host_SP1_sph[decoupled_idx].Fdx_da){
				printf("[DEBUG] SPH mismatched Fdx_da value coupled_idx=%d, decoupled_idx=%d, Fdx_da_coupled=%f, Fdx_da_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdx_da, host_SP1_sph[decoupled_idx].Fdx_da);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdy_da != host_SP1_sph[decoupled_idx].Fdy_da){
				printf("[DEBUG] SPH mismatched Fdy_da value coupled_idx=%d, decoupled_idx=%d, Fdy_da_coupled=%f, Fdy_da_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdy_da, host_SP1_sph[decoupled_idx].Fdy_da);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdz_da != host_SP1_sph[decoupled_idx].Fdz_da){
				printf("[DEBUG] SPH mismatched Fdz_da value coupled_idx=%d, decoupled_idx=%d, Fdz_da_coupled=%f, Fdz_da_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdz_da, host_SP1_sph[decoupled_idx].Fdz_da);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].DEMpor != host_SP1_sph[decoupled_idx].DEMpor){
				printf("[DEBUG] SPH mismatched DEMpor value coupled_idx=%d, decoupled_idx=%d, DEMpor_coupled=%f, DEMpor_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].DEMpor, host_SP1_sph[decoupled_idx].DEMpor);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].DEMvf != host_SP1_sph[decoupled_idx].DEMvf){
				printf("[DEBUG] SPH mismatched DEMvf value coupled_idx=%d, decoupled_idx=%d, DEMvf_coupled=%f, DEMvf_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].DEMvf, host_SP1_sph[decoupled_idx].DEMvf);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Q_sd != host_SP1_sph[decoupled_idx].Q_sd){
				printf("[DEBUG] SPH mismatched Q_sd value coupled_idx=%d, decoupled_idx=%d, Q_sd_coupled=%f, Q_sd_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Q_sd, host_SP1_sph[decoupled_idx].Q_sd);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Q_sdf != host_SP1_sph[decoupled_idx].Q_sdf){
				printf("[DEBUG] SPH mismatched Q_sdf value coupled_idx=%d, decoupled_idx=%d, Q_sdf_coupled=%f, Q_sdf_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Q_sdf, host_SP1_sph[decoupled_idx].Q_sdf);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Q_f != host_SP1_sph[decoupled_idx].Q_f){
				printf("[DEBUG] SPH mismatched Q_f value coupled_idx=%d, decoupled_idx=%d, Q_f_coupled=%f, Q_f_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Q_f, host_SP1_sph[decoupled_idx].Q_f);
				mismatch_count++;
			}
			for(int ci=0; ci<16; ci++){
				if(host_SP1[coupled_idx].ct_idx[ci] != host_SP1_sph[decoupled_idx].ct_idx[ci]){
					printf("[DEBUG] SPH mismatched ct_idx[%d] value coupled_idx=%d, decoupled_idx=%d, ct_idx_coupled=%d, ct_idx_decoupled=%d\n",
						ci, coupled_idx, decoupled_idx, host_SP1[coupled_idx].ct_idx[ci], host_SP1_sph[decoupled_idx].ct_idx[ci]);
					mismatch_count++;
				}
				for(int cj=0; cj<3; cj++){
					if(host_SP1[coupled_idx].del_s[ci][cj] != host_SP1_sph[decoupled_idx].del_s[ci][cj]){
						printf("[DEBUG] SPH mismatched del_s[%d][%d] value coupled_idx=%d, decoupled_idx=%d, del_s_coupled=%f, del_s_decoupled=%f\n",
							ci, cj, coupled_idx, decoupled_idx, host_SP1[coupled_idx].del_s[ci][cj], host_SP1_sph[decoupled_idx].del_s[ci][cj]);
						mismatch_count++;
					}
				}
			}
			if(host_SP1[coupled_idx].vol_power != host_SP1_sph[decoupled_idx].vol_power){
				printf("[DEBUG] SPH mismatched vol_power value coupled_idx=%d, decoupled_idx=%d, vol_power_coupled=%f, vol_power_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].vol_power, host_SP1_sph[decoupled_idx].vol_power);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].k_turb != host_SP1_sph[decoupled_idx].k_turb){
				printf("[DEBUG] SPH mismatched k_turb value coupled_idx=%d, decoupled_idx=%d, k_turb_coupled=%f, k_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].k_turb, host_SP1_sph[decoupled_idx].k_turb);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].e_turb != host_SP1_sph[decoupled_idx].e_turb){
				printf("[DEBUG] SPH mismatched e_turb value coupled_idx=%d, decoupled_idx=%d, e_turb_coupled=%f, e_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].e_turb, host_SP1_sph[decoupled_idx].e_turb);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres_ipp != host_SP1_sph[decoupled_idx].pres_ipp){
				printf("[DEBUG] SPH mismatched pres_ipp value coupled_idx=%d, decoupled_idx=%d, pres_ipp_coupled=%f, pres_ipp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres_ipp, host_SP1_sph[decoupled_idx].pres_ipp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres_ipp_p != host_SP1_sph[decoupled_idx].pres_ipp_p){
				printf("[DEBUG] SPH mismatched pres_ipp_p value coupled_idx=%d, decoupled_idx=%d, pres_ipp_p_coupled=%f, pres_ipp_p_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres_ipp_p, host_SP1_sph[decoupled_idx].pres_ipp_p);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres_ipp_s != host_SP1_sph[decoupled_idx].pres_ipp_s){
				printf("[DEBUG] SPH mismatched pres_ipp_s value coupled_idx=%d, decoupled_idx=%d, pres_ipp_s_coupled=%f, pres_ipp_s_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres_ipp_s, host_SP1_sph[decoupled_idx].pres_ipp_s);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test1 != host_SP1_sph[decoupled_idx].test1){
				printf("[DEBUG] SPH mismatched test1 value coupled_idx=%d, decoupled_idx=%d, test1_coupled=%f, test1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test1, host_SP1_sph[decoupled_idx].test1);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test2 != host_SP1_sph[decoupled_idx].test2){
				printf("[DEBUG] SPH mismatched test2 value coupled_idx=%d, decoupled_idx=%d, test2_coupled=%f, test2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test2, host_SP1_sph[decoupled_idx].test2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test3 != host_SP1_sph[decoupled_idx].test3){
				printf("[DEBUG] SPH mismatched test3 value coupled_idx=%d, decoupled_idx=%d, test3_coupled=%f, test3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test3, host_SP1_sph[decoupled_idx].test3);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test4 != host_SP1_sph[decoupled_idx].test4){
				printf("[DEBUG] SPH mismatched test4 value coupled_idx=%d, decoupled_idx=%d, test4_coupled=%f, test4_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test4, host_SP1_sph[decoupled_idx].test4);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test5 != host_SP1_sph[decoupled_idx].test5){
				printf("[DEBUG] SPH mismatched test5 value coupled_idx=%d, decoupled_idx=%d, test5_coupled=%f, test5_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test5, host_SP1_sph[decoupled_idx].test5);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test6 != host_SP1_sph[decoupled_idx].test6){
				printf("[DEBUG] SPH mismatched test6 value coupled_idx=%d, decoupled_idx=%d, test6_coupled=%f, test6_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test6, host_SP1_sph[decoupled_idx].test6);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test7 != host_SP1_sph[decoupled_idx].test7){
				printf("[DEBUG] SPH mismatched test7 value coupled_idx=%d, decoupled_idx=%d, test7_coupled=%f, test7_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test7, host_SP1_sph[decoupled_idx].test7);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test8 != host_SP1_sph[decoupled_idx].test8){
				printf("[DEBUG] SPH mismatched test8 value coupled_idx=%d, decoupled_idx=%d, test8_coupled=%f, test8_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test8, host_SP1_sph[decoupled_idx].test8);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].cond != host_SP1_sph[decoupled_idx].cond){
				printf("[DEBUG] SPH mismatched cond value coupled_idx=%d, decoupled_idx=%d, cond_coupled=%f, cond_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].cond, host_SP1_sph[decoupled_idx].cond);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].vol0 != host_SP1_sph[decoupled_idx].vol0){
				printf("[DEBUG] SPH mismatched vol0 value coupled_idx=%d, decoupled_idx=%d, vol0_coupled=%f, vol0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].vol0, host_SP1_sph[decoupled_idx].vol0);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].vol != host_SP1_sph[decoupled_idx].vol){
				printf("[DEBUG] SPH mismatched vol value coupled_idx=%d, decoupled_idx=%d, vol_coupled=%f, vol_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].vol, host_SP1_sph[decoupled_idx].vol);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].elix != host_SP1_sph[decoupled_idx].elix){
				printf("[DEBUG] SPH mismatched elix value coupled_idx=%d, decoupled_idx=%d, elix_coupled=%f, elix_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].elix, host_SP1_sph[decoupled_idx].elix);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].eliy != host_SP1_sph[decoupled_idx].eliy){
				printf("[DEBUG] SPH mismatched eliy value coupled_idx=%d, decoupled_idx=%d, eliy_coupled=%f, eliy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].eliy, host_SP1_sph[decoupled_idx].eliy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].eliz != host_SP1_sph[decoupled_idx].eliz){
				printf("[DEBUG] SPH mismatched eliz value coupled_idx=%d, decoupled_idx=%d, eliz_coupled=%f, eliz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].eliz, host_SP1_sph[decoupled_idx].eliz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fbx != host_SP1_sph[decoupled_idx].fbx){
				printf("[DEBUG] SPH mismatched fbx value coupled_idx=%d, decoupled_idx=%d, fbx_coupled=%f, fbx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fbx, host_SP1_sph[decoupled_idx].fbx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fby != host_SP1_sph[decoupled_idx].fby){
				printf("[DEBUG] SPH mismatched fby value coupled_idx=%d, decoupled_idx=%d, fby_coupled=%f, fby_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fby, host_SP1_sph[decoupled_idx].fby);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fbz != host_SP1_sph[decoupled_idx].fbz){
				printf("[DEBUG] SPH mismatched fbz value coupled_idx=%d, decoupled_idx=%d, fbz_coupled=%f, fbz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fbz, host_SP1_sph[decoupled_idx].fbz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fcx != host_SP1_sph[decoupled_idx].fcx){
				printf("[DEBUG] SPH mismatched fcx value coupled_idx=%d, decoupled_idx=%d, fcx_coupled=%f, fcx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fcx, host_SP1_sph[decoupled_idx].fcx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fcy != host_SP1_sph[decoupled_idx].fcy){
				printf("[DEBUG] SPH mismatched fcy value coupled_idx=%d, decoupled_idx=%d, fcy_coupled=%f, fcy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fcy, host_SP1_sph[decoupled_idx].fcy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fcz != host_SP1_sph[decoupled_idx].fcz){
				printf("[DEBUG] SPH mismatched fcz value coupled_idx=%d, decoupled_idx=%d, fcz_coupled=%f, fcz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fcz, host_SP1_sph[decoupled_idx].fcz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pos != host_SP1_sph[decoupled_idx].pos){
				printf("[DEBUG] SPH mismatched pos value coupled_idx=%d, decoupled_idx=%d, pos_coupled=%d, pos_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pos, host_SP1_sph[decoupled_idx].pos);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE1 != host_SP1_sph[decoupled_idx].PPE1){
				printf("[DEBUG] SPH mismatched PPE1 value coupled_idx=%d, decoupled_idx=%d, PPE1_coupled=%f, PPE1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE1, host_SP1_sph[decoupled_idx].PPE1);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE2 != host_SP1_sph[decoupled_idx].PPE2){
				printf("[DEBUG] SPH mismatched PPE2 value coupled_idx=%d, decoupled_idx=%d, PPE2_coupled=%f, PPE2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE2, host_SP1_sph[decoupled_idx].PPE2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE3 != host_SP1_sph[decoupled_idx].PPE3){
				printf("[DEBUG] SPH mismatched PPE3 value coupled_idx=%d, decoupled_idx=%d, PPE3_coupled=%f, PPE3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE3, host_SP1_sph[decoupled_idx].PPE3);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE4 != host_SP1_sph[decoupled_idx].PPE4){
				printf("[DEBUG] SPH mismatched PPE4 value coupled_idx=%d, decoupled_idx=%d, PPE4_coupled=%f, PPE4_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE4, host_SP1_sph[decoupled_idx].PPE4);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_ux != host_SP1_sph[decoupled_idx].XSPH_ux){
				printf("[DEBUG] SPH mismatched XSPH_ux value coupled_idx=%d, decoupled_idx=%d, XSPH_ux_coupled=%f, XSPH_ux_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_ux, host_SP1_sph[decoupled_idx].XSPH_ux);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_uy != host_SP1_sph[decoupled_idx].XSPH_uy){
				printf("[DEBUG] SPH mismatched XSPH_uy value coupled_idx=%d, decoupled_idx=%d, XSPH_uy_coupled=%f, XSPH_uy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_uy, host_SP1_sph[decoupled_idx].XSPH_uy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_uz != host_SP1_sph[decoupled_idx].XSPH_uz){
				printf("[DEBUG] SPH mismatched XSPH_uz value coupled_idx=%d, decoupled_idx=%d, XSPH_uz_coupled=%f, XSPH_uz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_uz, host_SP1_sph[decoupled_idx].XSPH_uz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_temp != host_SP1_sph[decoupled_idx].XSPH_temp){
				printf("[DEBUG] SPH mismatched XSPH_temp value coupled_idx=%d, decoupled_idx=%d, XSPH_temp_coupled=%f, XSPH_temp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_temp, host_SP1_sph[decoupled_idx].XSPH_temp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_m != host_SP1_sph[decoupled_idx].OpenBC_m){
				printf("[DEBUG] SPH mismatched OpenBC_m value coupled_idx=%d, decoupled_idx=%d, OpenBC_m_coupled=%f, OpenBC_m_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_m, host_SP1_sph[decoupled_idx].OpenBC_m);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_pres != host_SP1_sph[decoupled_idx].OpenBC_pres){
				printf("[DEBUG] SPH mismatched OpenBC_pres value coupled_idx=%d, decoupled_idx=%d, OpenBC_pres_coupled=%f, OpenBC_pres_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_pres, host_SP1_sph[decoupled_idx].OpenBC_pres);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_rho != host_SP1_sph[decoupled_idx].OpenBC_rho){
				printf("[DEBUG] SPH mismatched OpenBC_rho value coupled_idx=%d, decoupled_idx=%d, OpenBC_rho_coupled=%f, OpenBC_rho_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_rho, host_SP1_sph[decoupled_idx].OpenBC_rho);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_temp != host_SP1_sph[decoupled_idx].OpenBC_temp){
				printf("[DEBUG] SPH mismatched OpenBC_temp value coupled_idx=%d, decoupled_idx=%d, OpenBC_temp_coupled=%f, OpenBC_temp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_temp, host_SP1_sph[decoupled_idx].OpenBC_temp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_ux != host_SP1_sph[decoupled_idx].OpenBC_ux){
				printf("[DEBUG] SPH mismatched OpenBC_ux value coupled_idx=%d, decoupled_idx=%d, OpenBC_ux_coupled=%f, OpenBC_ux_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_ux, host_SP1_sph[decoupled_idx].OpenBC_ux);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_uy != host_SP1_sph[decoupled_idx].OpenBC_uy){
				printf("[DEBUG] SPH mismatched OpenBC_uy value coupled_idx=%d, decoupled_idx=%d, OpenBC_uy_coupled=%f, OpenBC_uy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_uy, host_SP1_sph[decoupled_idx].OpenBC_uy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_uz != host_SP1_sph[decoupled_idx].OpenBC_uz){
				printf("[DEBUG] SPH mismatched OpenBC_uz value coupled_idx=%d, decoupled_idx=%d, OpenBC_uz_coupled=%f, OpenBC_uz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_uz, host_SP1_sph[decoupled_idx].OpenBC_uz);
				mismatch_count++;
			}

			if(host_SP2[coupled_idx].rho_ref != host_SP2_sph[decoupled_idx].rho_ref){
				printf("[DEBUG] SPH mismatched rho_ref value coupled_idx=%d, decoupled_idx=%d, rho_ref_coupled=%f, rho_ref_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].rho_ref, host_SP2_sph[decoupled_idx].rho_ref);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].SR != host_SP2_sph[decoupled_idx].SR){
				printf("[DEBUG] SPH mismatched SR value coupled_idx=%d, decoupled_idx=%d, SR_coupled=%f, SR_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].SR, host_SP2_sph[decoupled_idx].SR);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].x0 != host_SP2_sph[decoupled_idx].x0){
				printf("[DEBUG] SPH mismatched x0 value coupled_idx=%d, decoupled_idx=%d, x0_coupled=%f, x0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].x0, host_SP2_sph[decoupled_idx].x0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].y0 != host_SP2_sph[decoupled_idx].y0){
				printf("[DEBUG] SPH mismatched y0 value coupled_idx=%d, decoupled_idx=%d, y0_coupled=%f, y0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].y0, host_SP2_sph[decoupled_idx].y0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].z0 != host_SP2_sph[decoupled_idx].z0){
				printf("[DEBUG] SPH mismatched z0 value coupled_idx=%d, decoupled_idx=%d, z0_coupled=%f, z0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].z0, host_SP2_sph[decoupled_idx].z0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].ux0 != host_SP2_sph[decoupled_idx].ux0){
				printf("[DEBUG] SPH mismatched ux0 value coupled_idx=%d, decoupled_idx=%d, ux0_coupled=%f, ux0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].ux0, host_SP2_sph[decoupled_idx].ux0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].uy0 != host_SP2_sph[decoupled_idx].uy0){
				printf("[DEBUG] SPH mismatched uy0 value coupled_idx=%d, decoupled_idx=%d, uy0_coupled=%f, uy0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].uy0, host_SP2_sph[decoupled_idx].uy0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].uz0 != host_SP2_sph[decoupled_idx].uz0){
				printf("[DEBUG] SPH mismatched uz0 value coupled_idx=%d, decoupled_idx=%d, uz0_coupled=%f, uz0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].uz0, host_SP2_sph[decoupled_idx].uz0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].wx0 != host_SP2_sph[decoupled_idx].wx0){
				printf("[DEBUG] SPH mismatched wx0 value coupled_idx=%d, decoupled_idx=%d, wx0_coupled=%f, wx0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].wx0, host_SP2_sph[decoupled_idx].wx0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].wy0 != host_SP2_sph[decoupled_idx].wy0){
				printf("[DEBUG] SPH mismatched wy0 value coupled_idx=%d, decoupled_idx=%d, wy0_coupled=%f, wy0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].wy0, host_SP2_sph[decoupled_idx].wy0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].wz0 != host_SP2_sph[decoupled_idx].wz0){
				printf("[DEBUG] SPH mismatched wz0 value coupled_idx=%d, decoupled_idx=%d, wz0_coupled=%f, wz0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].wz0, host_SP2_sph[decoupled_idx].wz0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].rho0 != host_SP2_sph[decoupled_idx].rho0){
				printf("[DEBUG] SPH mismatched rho0 value coupled_idx=%d, decoupled_idx=%d, rho0_coupled=%f, rho0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].rho0, host_SP2_sph[decoupled_idx].rho0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].drho0 != host_SP2_sph[decoupled_idx].drho0){
				printf("[DEBUG] SPH mismatched drho0 value coupled_idx=%d, decoupled_idx=%d, drho0_coupled=%f, drho0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].drho0, host_SP2_sph[decoupled_idx].drho0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].rad0 != host_SP2_sph[decoupled_idx].rad0){
				printf("[DEBUG] SPH mismatched rad0 value coupled_idx=%d, decoupled_idx=%d, rad0_coupled=%f, rad0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].rad0, host_SP2_sph[decoupled_idx].rad0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].vol0 != host_SP2_sph[decoupled_idx].vol0){
				printf("[DEBUG] SPH mismatched vol0 value coupled_idx=%d, decoupled_idx=%d, vol0_coupled=%f, vol0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].vol0, host_SP2_sph[decoupled_idx].vol0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].dvol0 != host_SP2_sph[decoupled_idx].dvol0){
				printf("[DEBUG] SPH mismatched dvol0 value coupled_idx=%d, decoupled_idx=%d, dvol0_coupled=%f, dvol0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].dvol0, host_SP2_sph[decoupled_idx].dvol0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].concn0 != host_SP2_sph[decoupled_idx].concn0){
				printf("[DEBUG] SPH mismatched concn0 value coupled_idx=%d, decoupled_idx=%d, concn0_coupled=%f, concn0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].concn0, host_SP2_sph[decoupled_idx].concn0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].enthalpy0 != host_SP2_sph[decoupled_idx].enthalpy0){
				printf("[DEBUG] SPH mismatched enthalpy0 value coupled_idx=%d, decoupled_idx=%d, enthalpy0_coupled=%f, enthalpy0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].enthalpy0, host_SP2_sph[decoupled_idx].enthalpy0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp0 != host_SP2_sph[decoupled_idx].temp0){
				printf("[DEBUG] SPH mismatched temp0 value coupled_idx=%d, decoupled_idx=%d, temp0_coupled=%f, temp0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp0, host_SP2_sph[decoupled_idx].temp0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp10 != host_SP2_sph[decoupled_idx].temp10){
				printf("[DEBUG] SPH mismatched temp1 value coupled_idx=%d, decoupled_idx=%d, temp1_coupled=%f, temp1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp10, host_SP2_sph[decoupled_idx].temp10);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp20 != host_SP2_sph[decoupled_idx].temp20){
				printf("[DEBUG] SPH mismatched temp2 value coupled_idx=%d, decoupled_idx=%d, temp2_coupled=%f, temp2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp20, host_SP2_sph[decoupled_idx].temp20);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp30 != host_SP2_sph[decoupled_idx].temp30){
				printf("[DEBUG] SPH mismatched temp3 value coupled_idx=%d, decoupled_idx=%d, temp3_coupled=%f, temp3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp30, host_SP2_sph[decoupled_idx].temp30);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].drho != host_P3_sph[decoupled_idx].drho){
				printf("[DEBUG] SPH mismatched drho value coupled_idx=%d, decoupled_idx=%d, drho_coupled=%f, drho_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].drho, host_P3_sph[decoupled_idx].drho);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dconcn != host_P3_sph[decoupled_idx].dconcn){
				printf("[DEBUG] SPH mismatched dconcn value coupled_idx=%d, decoupled_idx=%d, dconcn_coupled=%f, dconcn_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dconcn, host_P3_sph[decoupled_idx].dconcn);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].denthalpy != host_P3_sph[decoupled_idx].denthalpy){
				printf("[DEBUG] SPH mismatched denthalpy value coupled_idx=%d, decoupled_idx=%d, denthalpy_coupled=%f, denthalpy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].denthalpy, host_P3_sph[decoupled_idx].denthalpy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].drad != host_P3_sph[decoupled_idx].drad){
				printf("[DEBUG] SPH mismatched drad value coupled_idx=%d, decoupled_idx=%d, drad_coupled=%f, drad_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].drad, host_P3_sph[decoupled_idx].drad);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp != host_P3_sph[decoupled_idx].dtemp){
				printf("[DEBUG] SPH mismatched dtemp value coupled_idx=%d, decoupled_idx=%d, dtemp_coupled=%f, dtemp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp, host_P3_sph[decoupled_idx].dtemp);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp1 != host_P3_sph[decoupled_idx].dtemp1){
				printf("[DEBUG] SPH mismatched dtemp1 value coupled_idx=%d, decoupled_idx=%d, dtemp1_coupled=%f, dtemp1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp1, host_P3_sph[decoupled_idx].dtemp1);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp2 != host_P3_sph[decoupled_idx].dtemp2){
				printf("[DEBUG] SPH mismatched dtemp2 value coupled_idx=%d, decoupled_idx=%d, dtemp2_coupled=%f, dtemp2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp2, host_P3_sph[decoupled_idx].dtemp2);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp3 != host_P3_sph[decoupled_idx].dtemp3){
				printf("[DEBUG] SPH mismatched dtemp3 value coupled_idx=%d, decoupled_idx=%d, dtemp3_coupled=%f, dtemp3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp3, host_P3_sph[decoupled_idx].dtemp3);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].torqx != host_P3_sph[decoupled_idx].torqx){
				printf("[DEBUG] SPH mismatched torqx value coupled_idx=%d, decoupled_idx=%d, torqx_coupled=%f, torqx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].torqx, host_P3_sph[decoupled_idx].torqx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].torqy != host_P3_sph[decoupled_idx].torqy){
				printf("[DEBUG] SPH mismatched torqy value coupled_idx=%d, decoupled_idx=%d, torqy_coupled=%f, torqy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].torqy, host_P3_sph[decoupled_idx].torqy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].torqz != host_P3_sph[decoupled_idx].torqz){
				printf("[DEBUG] SPH mismatched torqz value coupled_idx=%d, decoupled_idx=%d, torqz_coupled=%f, torqz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].torqz, host_P3_sph[decoupled_idx].torqz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotalx != host_P3_sph[decoupled_idx].ftotalx){
				printf("[DEBUG] SPH mismatched ftotalx value coupled_idx=%d, decoupled_idx=%d, ftotalx_coupled=%f, ftotalx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotalx, host_P3_sph[decoupled_idx].ftotalx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotaly != host_P3_sph[decoupled_idx].ftotaly){
				printf("[DEBUG] SPH mismatched ftotaly value coupled_idx=%d, decoupled_idx=%d, ftotaly_coupled=%f, ftotaly_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotaly, host_P3_sph[decoupled_idx].ftotaly);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotalz != host_P3_sph[decoupled_idx].ftotalz){
				printf("[DEBUG] SPH mismatched ftotalz value coupled_idx=%d, decoupled_idx=%d, ftotalz_coupled=%f, ftotalz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotalz, host_P3_sph[decoupled_idx].ftotalz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotal != host_P3_sph[decoupled_idx].ftotal){
				printf("[DEBUG] SPH mismatched ftotal value coupled_idx=%d, decoupled_idx=%d, ftotal_coupled=%f, ftotal_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotal, host_P3_sph[decoupled_idx].ftotal);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fpx != host_P3_sph[decoupled_idx].fpx){
				printf("[DEBUG] SPH mismatched fpx value coupled_idx=%d, decoupled_idx=%d, fpx_coupled=%f, fpx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fpx, host_P3_sph[decoupled_idx].fpx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fpy != host_P3_sph[decoupled_idx].fpy){
				printf("[DEBUG] SPH mismatched fpy value coupled_idx=%d, decoupled_idx=%d, fpy_coupled=%f, fpy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fpy, host_P3_sph[decoupled_idx].fpy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fpz != host_P3_sph[decoupled_idx].fpz){
				printf("[DEBUG] SPH mismatched fpz value coupled_idx=%d, decoupled_idx=%d, fpz_coupled=%f, fpz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fpz, host_P3_sph[decoupled_idx].fpz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].vis_t != host_P3_sph[decoupled_idx].vis_t){
				printf("[DEBUG] SPH mismatched vis_t value coupled_idx=%d, decoupled_idx=%d, vis_t_coupled=%f, vis_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].vis_t, host_P3_sph[decoupled_idx].vis_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Sxx != host_P3_sph[decoupled_idx].Sxx){
				printf("[DEBUG] SPH mismatched Sxx value coupled_idx=%d, decoupled_idx=%d, Sxx_coupled=%f, Sxx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Sxx, host_P3_sph[decoupled_idx].Sxx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Sxy != host_P3_sph[decoupled_idx].Sxy){
				printf("[DEBUG] SPH mismatched Sxy value coupled_idx=%d, decoupled_idx=%d, Sxy_coupled=%f, Sxy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Sxy, host_P3_sph[decoupled_idx].Sxy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Sxz != host_P3_sph[decoupled_idx].Sxz){
				printf("[DEBUG] SPH mismatched Sxz value coupled_idx=%d, decoupled_idx=%d, Sxz_coupled=%f, Sxz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Sxz, host_P3_sph[decoupled_idx].Sxz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Syy != host_P3_sph[decoupled_idx].Syy){
				printf("[DEBUG] SPH mismatched Syy value coupled_idx=%d, decoupled_idx=%d, Syy_coupled=%f, Syy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Syy, host_P3_sph[decoupled_idx].Syy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Syz != host_P3_sph[decoupled_idx].Syz){
				printf("[DEBUG] SPH mismatched Syz value coupled_idx=%d, decoupled_idx=%d, Syz_coupled=%f, Syz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Syz, host_P3_sph[decoupled_idx].Syz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Szz != host_P3_sph[decoupled_idx].Szz){
				printf("[DEBUG] SPH mismatched Szz value coupled_idx=%d, decoupled_idx=%d, Szz_coupled=%f, Szz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Szz, host_P3_sph[decoupled_idx].Szz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dk_turb != host_P3_sph[decoupled_idx].dk_turb){
				printf("[DEBUG] SPH mismatched dk_turb value coupled_idx=%d, decoupled_idx=%d, dk_turb_coupled=%f, dk_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dk_turb, host_P3_sph[decoupled_idx].dk_turb);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].de_turb != host_P3_sph[decoupled_idx].de_turb){
				printf("[DEBUG] SPH mismatched de_turb value coupled_idx=%d, decoupled_idx=%d, de_turb_coupled=%f, de_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].de_turb, host_P3_sph[decoupled_idx].de_turb);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].lbl_surf != host_P3_sph[decoupled_idx].lbl_surf){
				printf("[DEBUG] SPH mismatched lbl_surf value coupled_idx=%d, decoupled_idx=%d, lbl_surf_coupled=%f, lbl_surf_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].lbl_surf, host_P3_sph[decoupled_idx].lbl_surf);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].cc != host_P3_sph[decoupled_idx].cc){
				printf("[DEBUG] SPH mismatched cc value coupled_idx=%d, decoupled_idx=%d, cc_coupled=%f, cc_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].cc, host_P3_sph[decoupled_idx].cc);
				mismatch_count++;
			}
			// Correction matrix Cm
			for(int ci=0; ci<Correction_Matrix_Size; ci++){
				for(int cj=0; cj<Correction_Matrix_Size; cj++){
					if(host_P3[coupled_idx].Cm[ci][cj] != host_P3_sph[decoupled_idx].Cm[ci][cj]){
						printf("[DEBUG] SPH mismatched Cm[%d][%d] value coupled_idx=%d, decoupled_idx=%d, Cm_coupled=%f, Cm_decoupled=%f\n",
							ci, cj, coupled_idx, decoupled_idx, host_P3[coupled_idx].Cm[ci][cj], host_P3_sph[decoupled_idx].Cm[ci][cj]);
						mismatch_count++;
					}
				}
			}
			if(host_P3[coupled_idx].nx != host_P3_sph[decoupled_idx].nx){
				printf("[DEBUG] SPH mismatched nx value coupled_idx=%d, decoupled_idx=%d, nx_coupled=%f, nx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx, host_P3_sph[decoupled_idx].nx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny != host_P3_sph[decoupled_idx].ny){
				printf("[DEBUG] SPH mismatched ny value coupled_idx=%d, decoupled_idx=%d, ny_coupled=%f, ny_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny, host_P3_sph[decoupled_idx].ny);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz != host_P3_sph[decoupled_idx].nz){
				printf("[DEBUG] SPH mismatched nz value coupled_idx=%d, decoupled_idx=%d, nz_coupled=%f, nz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz, host_P3_sph[decoupled_idx].nz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_u != host_P3_sph[decoupled_idx].nx_u){
				printf("[DEBUG] SPH mismatched nx_u value coupled_idx=%d, decoupled_idx=%d, nx_u_coupled=%f, nx_u_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_u, host_P3_sph[decoupled_idx].nx_u);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_u != host_P3_sph[decoupled_idx].ny_u){
				printf("[DEBUG] SPH mismatched ny_u value coupled_idx=%d, decoupled_idx=%d, ny_u_coupled=%f, ny_u_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_u, host_P3_sph[decoupled_idx].ny_u);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_u != host_P3_sph[decoupled_idx].nz_u){
				printf("[DEBUG] SPH mismatched nz_u value coupled_idx=%d, decoupled_idx=%d, nz_u_coupled=%f, nz_u_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_u, host_P3_sph[decoupled_idx].nz_u);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_w != host_P3_sph[decoupled_idx].nx_w){
				printf("[DEBUG] SPH mismatched nx_w value coupled_idx=%d, decoupled_idx=%d, nx_w_coupled=%f, nx_w_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_w, host_P3_sph[decoupled_idx].nx_w);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_w != host_P3_sph[decoupled_idx].ny_w){
				printf("[DEBUG] SPH mismatched ny_w value coupled_idx=%d, decoupled_idx=%d, ny_w_coupled=%f, ny_w_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_w, host_P3_sph[decoupled_idx].ny_w);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_w != host_P3_sph[decoupled_idx].nz_w){
				printf("[DEBUG] SPH mismatched nz_w value coupled_idx=%d, decoupled_idx=%d, nz_w_coupled=%f, nz_w_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_w, host_P3_sph[decoupled_idx].nz_w);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_t != host_P3_sph[decoupled_idx].nx_t){
				printf("[DEBUG] SPH mismatched nx_t value coupled_idx=%d, decoupled_idx=%d, nx_t_coupled=%f, nx_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_t, host_P3_sph[decoupled_idx].nx_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_t != host_P3_sph[decoupled_idx].ny_t){
				printf("[DEBUG] SPH mismatched ny_t value coupled_idx=%d, decoupled_idx=%d, ny_t_coupled=%f, ny_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_t, host_P3_sph[decoupled_idx].ny_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_t != host_P3_sph[decoupled_idx].nz_t){
				printf("[DEBUG] SPH mismatched nz_t value coupled_idx=%d, decoupled_idx=%d, nz_t_coupled=%f, nz_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_t, host_P3_sph[decoupled_idx].nz_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_tl != host_P3_sph[decoupled_idx].nx_tl){
				printf("[DEBUG] SPH mismatched nx_tl value coupled_idx=%d, decoupled_idx=%d, nx_tl_coupled=%f, nx_tl_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_tl, host_P3_sph[decoupled_idx].nx_tl);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_tl != host_P3_sph[decoupled_idx].ny_tl){
				printf("[DEBUG] SPH mismatched ny_tl value coupled_idx=%d, decoupled_idx=%d, ny_tl_coupled=%f, ny_tl_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_tl, host_P3_sph[decoupled_idx].ny_tl);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_tl != host_P3_sph[decoupled_idx].nz_tl){
				printf("[DEBUG] SPH mismatched nz_tl value coupled_idx=%d, decoupled_idx=%d, nz_tl_coupled=%f, nz_tl_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_tl, host_P3_sph[decoupled_idx].nz_tl);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_c != host_P3_sph[decoupled_idx].nx_c){
				printf("[DEBUG] SPH mismatched nx_c value coupled_idx=%d, decoupled_idx=%d, nx_c_coupled=%f, nx_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_c, host_P3_sph[decoupled_idx].nx_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_c != host_P3_sph[decoupled_idx].ny_c){
				printf("[DEBUG] SPH mismatched ny_c value coupled_idx=%d, decoupled_idx=%d, ny_c_coupled=%f, ny_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_c, host_P3_sph[decoupled_idx].ny_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_c != host_P3_sph[decoupled_idx].nz_c){
				printf("[DEBUG] SPH mismatched nz_c value coupled_idx=%d, decoupled_idx=%d, nz_c_coupled=%f, nz_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_c, host_P3_sph[decoupled_idx].nz_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nmag != host_P3_sph[decoupled_idx].nmag){
				printf("[DEBUG] SPH mismatched nmag value coupled_idx=%d, decoupled_idx=%d, nmag_coupled=%f, nmag_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nmag, host_P3_sph[decoupled_idx].nmag);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nmag_c != host_P3_sph[decoupled_idx].nmag_c){
				printf("[DEBUG] SPH mismatched nmag_c value coupled_idx=%d, decoupled_idx=%d, nmag_c_coupled=%f, nmag_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nmag_c, host_P3_sph[decoupled_idx].nmag_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].curv != host_P3_sph[decoupled_idx].curv){
				printf("[DEBUG] SPH mismatched curv value coupled_idx=%d, decoupled_idx=%d, curv_coupled=%f, curv_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].curv, host_P3_sph[decoupled_idx].curv);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fsx != host_P3_sph[decoupled_idx].fsx){
				printf("[DEBUG] SPH mismatched fsx value coupled_idx=%d, decoupled_idx=%d, fsx_coupled=%f, fsx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fsx, host_P3_sph[decoupled_idx].fsx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fsy != host_P3_sph[decoupled_idx].fsy){
				printf("[DEBUG] SPH mismatched fsy value coupled_idx=%d, decoupled_idx=%d, fsy_coupled=%f, fsy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fsy, host_P3_sph[decoupled_idx].fsy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fsz != host_P3_sph[decoupled_idx].fsz){
				printf("[DEBUG] SPH mismatched fsz value coupled_idx=%d, decoupled_idx=%d, fsz_coupled=%f, fsz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fsz, host_P3_sph[decoupled_idx].fsz);
				mismatch_count++;
			}
			decoupled_idx++;
			if (mismatch_count > 0) {
				printf("[ERROR] SPH mismatch detected. Exiting.\n");
				exit(EXIT_FAILURE);
			}
		}
		coupled_idx++;
	}
	printf("[DEBUG] SPH Total mismatch count: %d\n", mismatch_count);
	free(host_SP1);
	free(host_SP1_sph);
}

void Coupled_Decoupled_check_dem(part1 *dev_SP1, part1 *dev_SP1_dem, part2 *dev_SP2, part2 *dev_SP2_dem, part3 *dev_P3, part3 *dev_P3_dem, int_t *g_idx_in, int_t *g_idx_in_dem)
{
	part1 *host_SP1 = (part1*)malloc(sizeof(part1)*num_part2);
	part2 *host_SP2 = (part2*)malloc(sizeof(part2)*num_part2);
	part3 *host_P3 = (part3*)malloc(sizeof(part3)*num_part2);
	part1 *host_SP1_dem = (part1*)malloc(sizeof(part1)*num_part2_dem);
	part2 *host_SP2_dem = (part2*)malloc(sizeof(part2)*num_part2_dem);
	part3 *host_P3_dem = (part3*)malloc(sizeof(part3)*num_part2_dem);
	int_t *host_g_idx_in = (int_t*)malloc(sizeof(int_t)*num_part2);
	int_t *host_g_idx_in_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
	CUDA_CHECK(cudaMemcpy(host_SP1, dev_SP1, sizeof(part1)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_SP2, dev_SP2, sizeof(part2)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_P3, dev_P3, sizeof(part3)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_SP1_dem, dev_SP1_dem, sizeof(part1)*num_part2_dem, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_SP2_dem, dev_SP2_dem, sizeof(part2)*num_part2_dem, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_P3_dem, dev_P3_dem, sizeof(part3)*num_part2_dem, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_g_idx_in, g_idx_in, sizeof(int_t)*num_part2, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host_g_idx_in_dem, g_idx_in_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());
	int mismatch_count = 0;
	int coupled_idx = 0;
	int decoupled_idx = 0;
	for(int i=0; i<num_part2; i++) {
		if(host_SP1[i].p_type > 1000) {
			if(host_g_idx_in[coupled_idx] != host_g_idx_in_dem[decoupled_idx]){
				printf("[DEBUG] DEM mismatched g_idx_in value coupled_idx=%d, decoupled_idx=%d, g_idx_in_coupled=%d, g_idx_in_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_g_idx_in[coupled_idx], host_g_idx_in_dem[decoupled_idx]);
				mismatch_count++;
			}	
			if(host_SP1[coupled_idx].i_type != host_SP1_dem[decoupled_idx].i_type){
				printf("[DEBUG] DEM mismatched i_type value coupled_idx=%d, decoupled_idx=%d, i_type_coupled=%d, i_type_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].i_type, host_SP1_dem[decoupled_idx].i_type);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].buffer_type != host_SP1_dem[decoupled_idx].buffer_type){
				printf("[DEBUG] DEM mismatched buffer_type value coupled_idx=%d, decoupled_idx=%d, buffer_type_coupled=%d, buffer_type_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].buffer_type, host_SP1_dem[decoupled_idx].buffer_type);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].p_type != host_SP1_dem[decoupled_idx].p_type){
				printf("[DEBUG] DEM mismatched p_type value coupled_idx=%d, decoupled_idx=%d, p_type_coupled=%d, p_type_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].p_type, host_SP1_dem[decoupled_idx].p_type);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].dem_idx != host_SP1_dem[decoupled_idx].dem_idx){
				printf("[DEBUG] DEM mismatched dem_idx value coupled_idx=%d, decoupled_idx=%d, dem_idx_coupled=%d, dem_idx_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].dem_idx, host_SP1_dem[decoupled_idx].dem_idx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].ct_boundary != host_SP1_dem[decoupled_idx].ct_boundary){
				printf("[DEBUG] DEM mismatched ct_boundary value coupled_idx=%d, decoupled_idx=%d, ct_boundary_coupled=%d, ct_boundary_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].ct_boundary, host_SP1_dem[decoupled_idx].ct_boundary);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].x != host_SP1_dem[decoupled_idx].x){
				printf("[DEBUG] DEM mismatched x value coupled_idx=%d, decoupled_idx=%d, x_coupled=%f, x_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].x, host_SP1_dem[decoupled_idx].x);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].y != host_SP1_dem[decoupled_idx].y){
				printf("[DEBUG] DEM mismatched y value coupled_idx=%d, decoupled_idx=%d, y_coupled=%f, y_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].y, host_SP1_dem[decoupled_idx].y);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].z != host_SP1_dem[decoupled_idx].z){
				printf("[DEBUG] DEM mismatched z value coupled_idx=%d, decoupled_idx=%d, z_coupled=%f, z_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].z, host_SP1_dem[decoupled_idx].z);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].ux != host_SP1_dem[decoupled_idx].ux){
				printf("[DEBUG] DEM mismatched ux value coupled_idx=%d, decoupled_idx=%d, ux_coupled=%f, ux_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].ux, host_SP1_dem[decoupled_idx].ux);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].uy != host_SP1_dem[decoupled_idx].uy){
				printf("[DEBUG] DEM mismatched uy value coupled_idx=%d, decoupled_idx=%d, uy_coupled=%f, uy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].uy, host_SP1_dem[decoupled_idx].uy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].uz != host_SP1_dem[decoupled_idx].uz){
				printf("[DEBUG] DEM mismatched uz value coupled_idx=%d, decoupled_idx=%d, uz_coupled=%f, uz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].uz, host_SP1_dem[decoupled_idx].uz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].wx != host_SP1_dem[decoupled_idx].wx){
				printf("[DEBUG] DEM mismatched wx value coupled_idx=%d, decoupled_idx=%d, wx_coupled=%f, wx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].wx, host_SP1_dem[decoupled_idx].wx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].wy != host_SP1_dem[decoupled_idx].wy){
				printf("[DEBUG] DEM mismatched wy value coupled_idx=%d, decoupled_idx=%d, wy_coupled=%f, wy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].wy, host_SP1_dem[decoupled_idx].wy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].wz != host_SP1_dem[decoupled_idx].wz){
				printf("[DEBUG] DEM mismatched wz value coupled_idx=%d, decoupled_idx=%d, wz_coupled=%f, wz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].wz, host_SP1_dem[decoupled_idx].wz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].m != host_SP1_dem[decoupled_idx].m){
				printf("[DEBUG] DEM mismatched m value coupled_idx=%d, decoupled_idx=%d, m_coupled=%f, m_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].m, host_SP1_dem[decoupled_idx].m);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].ri != host_SP1_dem[decoupled_idx].ri){
				printf("[DEBUG] DEM mismatched ri value coupled_idx=%d, decoupled_idx=%d, ri_coupled=%f, ri_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].ri, host_SP1_dem[decoupled_idx].ri);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].rad != host_SP1_dem[decoupled_idx].rad){
				printf("[DEBUG] DEM mismatched rad value coupled_idx=%d, decoupled_idx=%d, rad_coupled=%f, rad_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].rad, host_SP1_dem[decoupled_idx].rad);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].h != host_SP1_dem[decoupled_idx].h){
				printf("[DEBUG] DEM mismatched h value coupled_idx=%d, decoupled_idx=%d, h_coupled=%f, h_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].h, host_SP1_dem[decoupled_idx].h);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp != host_SP1_dem[decoupled_idx].temp){
				printf("[DEBUG] DEM mismatched temp value coupled_idx=%d, decoupled_idx=%d, temp_coupled=%f, temp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp, host_SP1_dem[decoupled_idx].temp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp1 != host_SP1_dem[decoupled_idx].temp1){
				printf("[DEBUG] DEM mismatched temp1 value coupled_idx=%d, decoupled_idx=%d, temp1_coupled=%f, temp1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp1, host_SP1_dem[decoupled_idx].temp1);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp2 != host_SP1_dem[decoupled_idx].temp2){
				printf("[DEBUG] DEM mismatched temp2 value coupled_idx=%d, decoupled_idx=%d, temp2_coupled=%f, temp2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp2, host_SP1_dem[decoupled_idx].temp2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].temp3 != host_SP1_dem[decoupled_idx].temp3){
				printf("[DEBUG] DEM mismatched temp3 value coupled_idx=%d, decoupled_idx=%d, temp3_coupled=%f, temp3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].temp3, host_SP1_dem[decoupled_idx].temp3);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres != host_SP1_dem[decoupled_idx].pres){
				printf("[DEBUG] DEM mismatched pres value coupled_idx=%d, decoupled_idx=%d, pres_coupled=%f, pres_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres, host_SP1_dem[decoupled_idx].pres);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].rho != host_SP1_dem[decoupled_idx].rho){
				printf("[DEBUG] DEM mismatched rho value coupled_idx=%d, decoupled_idx=%d, rho_coupled=%f, rho_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].rho, host_SP1_dem[decoupled_idx].rho);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].flt_s != host_SP1_dem[decoupled_idx].flt_s){
				printf("[DEBUG] DEM mismatched flt_s value coupled_idx=%d, decoupled_idx=%d, flt_s_coupled=%f, flt_s_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].flt_s, host_SP1_dem[decoupled_idx].flt_s);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].flt_sd != host_SP1_dem[decoupled_idx].flt_sd){
				printf("[DEBUG] DEM mismatched flt_sd value coupled_idx=%d, decoupled_idx=%d, flt_sd_coupled=%f, flt_sd_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].flt_sd, host_SP1_dem[decoupled_idx].flt_sd);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].flt_sd_2 != host_SP1_dem[decoupled_idx].flt_sd_2){
				printf("[DEBUG] DEM mismatched flt_sd_2 value coupled_idx=%d, decoupled_idx=%d, flt_sd_2_coupled=%f, flt_sd_2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].flt_sd_2, host_SP1_dem[decoupled_idx].flt_sd_2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].w_dx != host_SP1_dem[decoupled_idx].w_dx){
				printf("[DEBUG] DEM mismatched w_dx value coupled_idx=%d, decoupled_idx=%d, w_dx_coupled=%f, w_dx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].w_dx, host_SP1_dem[decoupled_idx].w_dx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].enthalpy != host_SP1_dem[decoupled_idx].enthalpy){
				printf("[DEBUG] DEM mismatched enthalpy value coupled_idx=%d, decoupled_idx=%d, enthalpy_coupled=%f, enthalpy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].enthalpy, host_SP1_dem[decoupled_idx].enthalpy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].concn != host_SP1_dem[decoupled_idx].concn){
				printf("[DEBUG] DEM mismatched concn value coupled_idx=%d, decoupled_idx=%d, concn_coupled=%f, concn_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].concn, host_SP1_dem[decoupled_idx].concn);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].grad_rhox != host_SP1_dem[decoupled_idx].grad_rhox){
				printf("[DEBUG] DEM mismatched grad_rhox value coupled_idx=%d, decoupled_idx=%d, grad_rhox_coupled=%f, grad_rhox_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].grad_rhox, host_SP1_dem[decoupled_idx].grad_rhox);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].grad_rhoy != host_SP1_dem[decoupled_idx].grad_rhoy){
				printf("[DEBUG] DEM mismatched grad_rhoy value coupled_idx=%d, decoupled_idx=%d, grad_rhoy_coupled=%f, grad_rhoy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].grad_rhoy, host_SP1_dem[decoupled_idx].grad_rhoy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].grad_rhoz != host_SP1_dem[decoupled_idx].grad_rhoz){
				printf("[DEBUG] DEM mismatched grad_rhoz value coupled_idx=%d, decoupled_idx=%d, grad_rhoz_coupled=%f, grad_rhoz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].grad_rhoz, host_SP1_dem[decoupled_idx].grad_rhoz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pgf_x != host_SP1_dem[decoupled_idx].pgf_x){
				printf("[DEBUG] DEM mismatched pgf_x value coupled_idx=%d, decoupled_idx=%d, pgf_x_coupled=%f, pgf_x_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pgf_x, host_SP1_dem[decoupled_idx].pgf_x);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pgf_y != host_SP1_dem[decoupled_idx].pgf_y){
				printf("[DEBUG] DEM mismatched pgf_y value coupled_idx=%d, decoupled_idx=%d, pgf_y_coupled=%f, pgf_y_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pgf_y, host_SP1_dem[decoupled_idx].pgf_y);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pgf_z != host_SP1_dem[decoupled_idx].pgf_z){
				printf("[DEBUG] DEM mismatched pgf_z value coupled_idx=%d, decoupled_idx=%d, pgf_z_coupled=%f, pgf_z_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pgf_z, host_SP1_dem[decoupled_idx].pgf_z);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdx_b != host_SP1_dem[decoupled_idx].Fdx_b){
				printf("[DEBUG] DEM mismatched Fdx_b value coupled_idx=%d, decoupled_idx=%d, Fdx_b_coupled=%f, Fdx_b_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdx_b, host_SP1_dem[decoupled_idx].Fdx_b);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdy_b != host_SP1_dem[decoupled_idx].Fdy_b){
				printf("[DEBUG] DEM mismatched Fdy_b value coupled_idx=%d, decoupled_idx=%d, Fdy_b_coupled=%f, Fdy_b_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdy_b, host_SP1_dem[decoupled_idx].Fdy_b);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdz_b != host_SP1_dem[decoupled_idx].Fdz_b){
				printf("[DEBUG] DEM mismatched Fdz_b value coupled_idx=%d, decoupled_idx=%d, Fdz_b_coupled=%f, Fdz_b_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdz_b, host_SP1_dem[decoupled_idx].Fdz_b);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdx_df != host_SP1_dem[decoupled_idx].Fdx_df){
				printf("[DEBUG] DEM mismatched Fdx_df value coupled_idx=%d, decoupled_idx=%d, Fdx_df_coupled=%f, Fdx_df_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdx_df, host_SP1_dem[decoupled_idx].Fdx_df);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdy_df != host_SP1_dem[decoupled_idx].Fdy_df){
				printf("[DEBUG] DEM mismatched Fdy_df value coupled_idx=%d, decoupled_idx=%d, Fdy_df_coupled=%f, Fdy_df_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdy_df, host_SP1_dem[decoupled_idx].Fdy_df);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdz_df != host_SP1_dem[decoupled_idx].Fdz_df){
				printf("[DEBUG] DEM mismatched Fdz_df value coupled_idx=%d, decoupled_idx=%d, Fdz_df_coupled=%f, Fdz_df_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdz_df, host_SP1_dem[decoupled_idx].Fdz_df);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdx_da != host_SP1_dem[decoupled_idx].Fdx_da){
				printf("[DEBUG] DEM mismatched Fdx_da value coupled_idx=%d, decoupled_idx=%d, Fdx_da_coupled=%f, Fdx_da_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdx_da, host_SP1_dem[decoupled_idx].Fdx_da);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdy_da != host_SP1_dem[decoupled_idx].Fdy_da){
				printf("[DEBUG] DEM mismatched Fdy_da value coupled_idx=%d, decoupled_idx=%d, Fdy_da_coupled=%f, Fdy_da_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdy_da, host_SP1_dem[decoupled_idx].Fdy_da);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Fdz_da != host_SP1_dem[decoupled_idx].Fdz_da){
				printf("[DEBUG] DEM mismatched Fdz_da value coupled_idx=%d, decoupled_idx=%d, Fdz_da_coupled=%f, Fdz_da_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Fdz_da, host_SP1_dem[decoupled_idx].Fdz_da);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].DEMpor != host_SP1_dem[decoupled_idx].DEMpor){
				printf("[DEBUG] DEM mismatched DEMpor value coupled_idx=%d, decoupled_idx=%d, DEMpor_coupled=%f, DEMpor_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].DEMpor, host_SP1_dem[decoupled_idx].DEMpor);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].DEMvf != host_SP1_dem[decoupled_idx].DEMvf){
				printf("[DEBUG] DEM mismatched DEMvf value coupled_idx=%d, decoupled_idx=%d, DEMvf_coupled=%f, DEMvf_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].DEMvf, host_SP1_dem[decoupled_idx].DEMvf);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Q_sd != host_SP1_dem[decoupled_idx].Q_sd){
				printf("[DEBUG] DEM mismatched Q_sd value coupled_idx=%d, decoupled_idx=%d, Q_sd_coupled=%f, Q_sd_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Q_sd, host_SP1_dem[decoupled_idx].Q_sd);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Q_sdf != host_SP1_dem[decoupled_idx].Q_sdf){
				printf("[DEBUG] DEM mismatched Q_sdf value coupled_idx=%d, decoupled_idx=%d, Q_sdf_coupled=%f, Q_sdf_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Q_sdf, host_SP1_dem[decoupled_idx].Q_sdf);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].Q_f != host_SP1_dem[decoupled_idx].Q_f){
				printf("[DEBUG] DEM mismatched Q_f value coupled_idx=%d, decoupled_idx=%d, Q_f_coupled=%f, Q_f_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].Q_f, host_SP1_dem[decoupled_idx].Q_f);
				mismatch_count++;
			}
			for(int ci=0; ci<16; ci++){
				if(host_SP1[coupled_idx].ct_idx[ci] != host_SP1_dem[decoupled_idx].ct_idx[ci]){
					printf("[DEBUG] DEM mismatched ct_idx[%d] value coupled_idx=%d, decoupled_idx=%d, ct_idx_coupled=%d, ct_idx_decoupled=%d\n",
						ci, coupled_idx, decoupled_idx, host_SP1[coupled_idx].ct_idx[ci], host_SP1_dem[decoupled_idx].ct_idx[ci]);
					mismatch_count++;
				}
				for(int cj=0; cj<3; cj++){
					if(host_SP1[coupled_idx].del_s[ci][cj] != host_SP1_dem[decoupled_idx].del_s[ci][cj]){
						printf("[DEBUG] DEM mismatched del_s[%d][%d] value coupled_idx=%d, decoupled_idx=%d, del_s_coupled=%f, del_s_decoupled=%f\n",
							ci, cj, coupled_idx, decoupled_idx, host_SP1[coupled_idx].del_s[ci][cj], host_SP1_dem[decoupled_idx].del_s[ci][cj]);
						mismatch_count++;
					}
				}
			}
			if(host_SP1[coupled_idx].vol_power != host_SP1_dem[decoupled_idx].vol_power){
				printf("[DEBUG] DEM mismatched vol_power value coupled_idx=%d, decoupled_idx=%d, vol_power_coupled=%f, vol_power_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].vol_power, host_SP1_dem[decoupled_idx].vol_power);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].k_turb != host_SP1_dem[decoupled_idx].k_turb){
				printf("[DEBUG] DEM mismatched k_turb value coupled_idx=%d, decoupled_idx=%d, k_turb_coupled=%f, k_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].k_turb, host_SP1_dem[decoupled_idx].k_turb);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].e_turb != host_SP1_dem[decoupled_idx].e_turb){
				printf("[DEBUG] DEM mismatched e_turb value coupled_idx=%d, decoupled_idx=%d, e_turb_coupled=%f, e_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].e_turb, host_SP1_dem[decoupled_idx].e_turb);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres_ipp != host_SP1_dem[decoupled_idx].pres_ipp){
				printf("[DEBUG] DEM mismatched pres_ipp value coupled_idx=%d, decoupled_idx=%d, pres_ipp_coupled=%f, pres_ipp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres_ipp, host_SP1_dem[decoupled_idx].pres_ipp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres_ipp_p != host_SP1_dem[decoupled_idx].pres_ipp_p){
				printf("[DEBUG] DEM mismatched pres_ipp_p value coupled_idx=%d, decoupled_idx=%d, pres_ipp_p_coupled=%f, pres_ipp_p_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres_ipp_p, host_SP1_dem[decoupled_idx].pres_ipp_p);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pres_ipp_s != host_SP1_dem[decoupled_idx].pres_ipp_s){
				printf("[DEBUG] DEM mismatched pres_ipp_s value coupled_idx=%d, decoupled_idx=%d, pres_ipp_s_coupled=%f, pres_ipp_s_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pres_ipp_s, host_SP1_dem[decoupled_idx].pres_ipp_s);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test1 != host_SP1_dem[decoupled_idx].test1){
				printf("[DEBUG] DEM mismatched test1 value coupled_idx=%d, decoupled_idx=%d, test1_coupled=%f, test1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test1, host_SP1_dem[decoupled_idx].test1);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test2 != host_SP1_dem[decoupled_idx].test2){
				printf("[DEBUG] DEM mismatched test2 value coupled_idx=%d, decoupled_idx=%d, test2_coupled=%f, test2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test2, host_SP1_dem[decoupled_idx].test2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test3 != host_SP1_dem[decoupled_idx].test3){
				printf("[DEBUG] DEM mismatched test3 value coupled_idx=%d, decoupled_idx=%d, test3_coupled=%f, test3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test3, host_SP1_dem[decoupled_idx].test3);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test4 != host_SP1_dem[decoupled_idx].test4){
				printf("[DEBUG] DEM mismatched test4 value coupled_idx=%d, decoupled_idx=%d, test4_coupled=%f, test4_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test4, host_SP1_dem[decoupled_idx].test4);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test5 != host_SP1_dem[decoupled_idx].test5){
				printf("[DEBUG] DEM mismatched test5 value coupled_idx=%d, decoupled_idx=%d, test5_coupled=%f, test5_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test5, host_SP1_dem[decoupled_idx].test5);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test6 != host_SP1_dem[decoupled_idx].test6){
				printf("[DEBUG] DEM mismatched test6 value coupled_idx=%d, decoupled_idx=%d, test6_coupled=%f, test6_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test6, host_SP1_dem[decoupled_idx].test6);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test7 != host_SP1_dem[decoupled_idx].test7){
				printf("[DEBUG] DEM mismatched test7 value coupled_idx=%d, decoupled_idx=%d, test7_coupled=%f, test7_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test7, host_SP1_dem[decoupled_idx].test7);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].test8 != host_SP1_dem[decoupled_idx].test8){
				printf("[DEBUG] DEM mismatched test8 value coupled_idx=%d, decoupled_idx=%d, test8_coupled=%f, test8_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].test8, host_SP1_dem[decoupled_idx].test8);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].cond != host_SP1_dem[decoupled_idx].cond){
				printf("[DEBUG] DEM mismatched cond value coupled_idx=%d, decoupled_idx=%d, cond_coupled=%f, cond_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].cond, host_SP1_dem[decoupled_idx].cond);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].vol0 != host_SP1_dem[decoupled_idx].vol0){
				printf("[DEBUG] DEM mismatched vol0 value coupled_idx=%d, decoupled_idx=%d, vol0_coupled=%f, vol0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].vol0, host_SP1_dem[decoupled_idx].vol0);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].vol != host_SP1_dem[decoupled_idx].vol){
				printf("[DEBUG] DEM mismatched vol value coupled_idx=%d, decoupled_idx=%d, vol_coupled=%f, vol_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].vol, host_SP1_dem[decoupled_idx].vol);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].elix != host_SP1_dem[decoupled_idx].elix){
				printf("[DEBUG] DEM mismatched elix value coupled_idx=%d, decoupled_idx=%d, elix_coupled=%f, elix_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].elix, host_SP1_dem[decoupled_idx].elix);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].eliy != host_SP1_dem[decoupled_idx].eliy){
				printf("[DEBUG] DEM mismatched eliy value coupled_idx=%d, decoupled_idx=%d, eliy_coupled=%f, eliy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].eliy, host_SP1_dem[decoupled_idx].eliy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].eliz != host_SP1_dem[decoupled_idx].eliz){
				printf("[DEBUG] DEM mismatched eliz value coupled_idx=%d, decoupled_idx=%d, eliz_coupled=%f, eliz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].eliz, host_SP1_dem[decoupled_idx].eliz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fbx != host_SP1_dem[decoupled_idx].fbx){
				printf("[DEBUG] DEM mismatched fbx value coupled_idx=%d, decoupled_idx=%d, fbx_coupled=%f, fbx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fbx, host_SP1_dem[decoupled_idx].fbx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fby != host_SP1_dem[decoupled_idx].fby){
				printf("[DEBUG] DEM mismatched fby value coupled_idx=%d, decoupled_idx=%d, fby_coupled=%f, fby_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fby, host_SP1_dem[decoupled_idx].fby);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fbz != host_SP1_dem[decoupled_idx].fbz){
				printf("[DEBUG] DEM mismatched fbz value coupled_idx=%d, decoupled_idx=%d, fbz_coupled=%f, fbz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fbz, host_SP1_dem[decoupled_idx].fbz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fcx != host_SP1_dem[decoupled_idx].fcx){
				printf("[DEBUG] DEM mismatched fcx value coupled_idx=%d, decoupled_idx=%d, fcx_coupled=%f, fcx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fcx, host_SP1_dem[decoupled_idx].fcx);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fcy != host_SP1_dem[decoupled_idx].fcy){
				printf("[DEBUG] DEM mismatched fcy value coupled_idx=%d, decoupled_idx=%d, fcy_coupled=%f, fcy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fcy, host_SP1_dem[decoupled_idx].fcy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].fcz != host_SP1_dem[decoupled_idx].fcz){
				printf("[DEBUG] DEM mismatched fcz value coupled_idx=%d, decoupled_idx=%d, fcz_coupled=%f, fcz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].fcz, host_SP1_dem[decoupled_idx].fcz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].pos != host_SP1_dem[decoupled_idx].pos){
				printf("[DEBUG] DEM mismatched pos value coupled_idx=%d, decoupled_idx=%d, pos_coupled=%d, pos_decoupled=%d\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].pos, host_SP1_dem[decoupled_idx].pos);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE1 != host_SP1_dem[decoupled_idx].PPE1){
				printf("[DEBUG] DEM mismatched PPE1 value coupled_idx=%d, decoupled_idx=%d, PPE1_coupled=%f, PPE1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE1, host_SP1_dem[decoupled_idx].PPE1);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE2 != host_SP1_dem[decoupled_idx].PPE2){
				printf("[DEBUG] DEM mismatched PPE2 value coupled_idx=%d, decoupled_idx=%d, PPE2_coupled=%f, PPE2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE2, host_SP1_dem[decoupled_idx].PPE2);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE3 != host_SP1_dem[decoupled_idx].PPE3){
				printf("[DEBUG] DEM mismatched PPE3 value coupled_idx=%d, decoupled_idx=%d, PPE3_coupled=%f, PPE3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE3, host_SP1_dem[decoupled_idx].PPE3);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].PPE4 != host_SP1_dem[decoupled_idx].PPE4){
				printf("[DEBUG] DEM mismatched PPE4 value coupled_idx=%d, decoupled_idx=%d, PPE4_coupled=%f, PPE4_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].PPE4, host_SP1_dem[decoupled_idx].PPE4);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_ux != host_SP1_dem[decoupled_idx].XSPH_ux){
				printf("[DEBUG] DEM mismatched XSPH_ux value coupled_idx=%d, decoupled_idx=%d, XSPH_ux_coupled=%f, XSPH_ux_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_ux, host_SP1_dem[decoupled_idx].XSPH_ux);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_uy != host_SP1_dem[decoupled_idx].XSPH_uy){
				printf("[DEBUG] DEM mismatched XSPH_uy value coupled_idx=%d, decoupled_idx=%d, XSPH_uy_coupled=%f, XSPH_uy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_uy, host_SP1_dem[decoupled_idx].XSPH_uy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_uz != host_SP1_dem[decoupled_idx].XSPH_uz){
				printf("[DEBUG] DEM mismatched XSPH_uz value coupled_idx=%d, decoupled_idx=%d, XSPH_uz_coupled=%f, XSPH_uz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_uz, host_SP1_dem[decoupled_idx].XSPH_uz);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].XSPH_temp != host_SP1_dem[decoupled_idx].XSPH_temp){
				printf("[DEBUG] DEM mismatched XSPH_temp value coupled_idx=%d, decoupled_idx=%d, XSPH_temp_coupled=%f, XSPH_temp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].XSPH_temp, host_SP1_dem[decoupled_idx].XSPH_temp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_m != host_SP1_dem[decoupled_idx].OpenBC_m){
				printf("[DEBUG] DEM mismatched OpenBC_m value coupled_idx=%d, decoupled_idx=%d, OpenBC_m_coupled=%f, OpenBC_m_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_m, host_SP1_dem[decoupled_idx].OpenBC_m);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_pres != host_SP1_dem[decoupled_idx].OpenBC_pres){
				printf("[DEBUG] DEM mismatched OpenBC_pres value coupled_idx=%d, decoupled_idx=%d, OpenBC_pres_coupled=%f, OpenBC_pres_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_pres, host_SP1_dem[decoupled_idx].OpenBC_pres);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_rho != host_SP1_dem[decoupled_idx].OpenBC_rho){
				printf("[DEBUG] DEM mismatched OpenBC_rho value coupled_idx=%d, decoupled_idx=%d, OpenBC_rho_coupled=%f, OpenBC_rho_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_rho, host_SP1_dem[decoupled_idx].OpenBC_rho);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_temp != host_SP1_dem[decoupled_idx].OpenBC_temp){
				printf("[DEBUG] DEM mismatched OpenBC_temp value coupled_idx=%d, decoupled_idx=%d, OpenBC_temp_coupled=%f, OpenBC_temp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_temp, host_SP1_dem[decoupled_idx].OpenBC_temp);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_ux != host_SP1_dem[decoupled_idx].OpenBC_ux){
				printf("[DEBUG] DEM mismatched OpenBC_ux value coupled_idx=%d, decoupled_idx=%d, OpenBC_ux_coupled=%f, OpenBC_ux_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_ux, host_SP1_dem[decoupled_idx].OpenBC_ux);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_uy != host_SP1_dem[decoupled_idx].OpenBC_uy){
				printf("[DEBUG] DEM mismatched OpenBC_uy value coupled_idx=%d, decoupled_idx=%d, OpenBC_uy_coupled=%f, OpenBC_uy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_uy, host_SP1_dem[decoupled_idx].OpenBC_uy);
				mismatch_count++;
			}
			if(host_SP1[coupled_idx].OpenBC_uz != host_SP1_dem[decoupled_idx].OpenBC_uz){
				printf("[DEBUG] DEM mismatched OpenBC_uz value coupled_idx=%d, decoupled_idx=%d, OpenBC_uz_coupled=%f, OpenBC_uz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP1[coupled_idx].OpenBC_uz, host_SP1_dem[decoupled_idx].OpenBC_uz);
				mismatch_count++;
			}
			
			if(host_SP2[coupled_idx].rho_ref != host_SP2_dem[decoupled_idx].rho_ref){
				printf("[DEBUG] DEM mismatched rho_ref value coupled_idx=%d, decoupled_idx=%d, rho_ref_coupled=%f, rho_ref_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].rho_ref, host_SP2_dem[decoupled_idx].rho_ref);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].SR != host_SP2_dem[decoupled_idx].SR){
				printf("[DEBUG] DEM mismatched SR value coupled_idx=%d, decoupled_idx=%d, SR_coupled=%f, SR_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].SR, host_SP2_dem[decoupled_idx].SR);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].x0 != host_SP2_dem[decoupled_idx].x0){
				printf("[DEBUG] DEM mismatched x0 value coupled_idx=%d, decoupled_idx=%d, x0_coupled=%f, x0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].x0, host_SP2_dem[decoupled_idx].x0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].y0 != host_SP2_dem[decoupled_idx].y0){
				printf("[DEBUG] DEM mismatched y0 value coupled_idx=%d, decoupled_idx=%d, y0_coupled=%f, y0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].y0, host_SP2_dem[decoupled_idx].y0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].z0 != host_SP2_dem[decoupled_idx].z0){
				printf("[DEBUG] DEM mismatched z0 value coupled_idx=%d, decoupled_idx=%d, z0_coupled=%f, z0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].z0, host_SP2_dem[decoupled_idx].z0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].ux0 != host_SP2_dem[decoupled_idx].ux0){
				printf("[DEBUG] DEM mismatched ux0 value coupled_idx=%d, decoupled_idx=%d, ux0_coupled=%f, ux0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].ux0, host_SP2_dem[decoupled_idx].ux0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].uy0 != host_SP2_dem[decoupled_idx].uy0){
				printf("[DEBUG] DEM mismatched uy0 value coupled_idx=%d, decoupled_idx=%d, uy0_coupled=%f, uy0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].uy0, host_SP2_dem[decoupled_idx].uy0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].uz0 != host_SP2_dem[decoupled_idx].uz0){
				printf("[DEBUG] DEM mismatched uz0 value coupled_idx=%d, decoupled_idx=%d, uz0_coupled=%f, uz0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].uz0, host_SP2_dem[decoupled_idx].uz0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].wx0 != host_SP2_dem[decoupled_idx].wx0){
				printf("[DEBUG] DEM mismatched wx0 value coupled_idx=%d, decoupled_idx=%d, wx0_coupled=%f, wx0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].wx0, host_SP2_dem[decoupled_idx].wx0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].wy0 != host_SP2_dem[decoupled_idx].wy0){
				printf("[DEBUG] DEM mismatched wy0 value coupled_idx=%d, decoupled_idx=%d, wy0_coupled=%f, wy0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].wy0, host_SP2_dem[decoupled_idx].wy0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].wz0 != host_SP2_dem[decoupled_idx].wz0){
				printf("[DEBUG] DEM mismatched wz0 value coupled_idx=%d, decoupled_idx=%d, wz0_coupled=%f, wz0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].wz0, host_SP2_dem[decoupled_idx].wz0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].rho0 != host_SP2_dem[decoupled_idx].rho0){
				printf("[DEBUG] DEM mismatched rho0 value coupled_idx=%d, decoupled_idx=%d, rho0_coupled=%f, rho0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].rho0, host_SP2_dem[decoupled_idx].rho0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].drho0 != host_SP2_dem[decoupled_idx].drho0){
				printf("[DEBUG] DEM mismatched drho0 value coupled_idx=%d, decoupled_idx=%d, drho0_coupled=%f, drho0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].drho0, host_SP2_dem[decoupled_idx].drho0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].rad0 != host_SP2_dem[decoupled_idx].rad0){
				printf("[DEBUG] DEM mismatched rad0 value coupled_idx=%d, decoupled_idx=%d, rad0_coupled=%f, rad0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].rad0, host_SP2_dem[decoupled_idx].rad0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].vol0 != host_SP2_dem[decoupled_idx].vol0){
				printf("[DEBUG] DEM mismatched vol0 value coupled_idx=%d, decoupled_idx=%d, vol0_coupled=%f, vol0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].vol0, host_SP2_dem[decoupled_idx].vol0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].dvol0 != host_SP2_dem[decoupled_idx].dvol0){
				printf("[DEBUG] DEM mismatched dvol0 value coupled_idx=%d, decoupled_idx=%d, dvol0_coupled=%f, dvol0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].dvol0, host_SP2_dem[decoupled_idx].dvol0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].concn0 != host_SP2_dem[decoupled_idx].concn0){
				printf("[DEBUG] DEM mismatched concn0 value coupled_idx=%d, decoupled_idx=%d, concn0_coupled=%f, concn0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].concn0, host_SP2_dem[decoupled_idx].concn0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].enthalpy0 != host_SP2_dem[decoupled_idx].enthalpy0){
				printf("[DEBUG] DEM mismatched enthalpy0 value coupled_idx=%d, decoupled_idx=%d, enthalpy0_coupled=%f, enthalpy0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].enthalpy0, host_SP2_dem[decoupled_idx].enthalpy0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp0 != host_SP2_dem[decoupled_idx].temp0){
				printf("[DEBUG] DEM mismatched temp0 value coupled_idx=%d, decoupled_idx=%d, temp0_coupled=%f, temp0_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp0, host_SP2_dem[decoupled_idx].temp0);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp10 != host_SP2_dem[decoupled_idx].temp10){
				printf("[DEBUG] DEM mismatched temp1 value coupled_idx=%d, decoupled_idx=%d, temp1_coupled=%f, temp1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp10, host_SP2_dem[decoupled_idx].temp10);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp20 != host_SP2_dem[decoupled_idx].temp20){
				printf("[DEBUG] DEM mismatched temp2 value coupled_idx=%d, decoupled_idx=%d, temp2_coupled=%f, temp2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp20, host_SP2_dem[decoupled_idx].temp20);
				mismatch_count++;
			}
			if(host_SP2[coupled_idx].temp30 != host_SP2_dem[decoupled_idx].temp30){
				printf("[DEBUG] DEM mismatched temp3 value coupled_idx=%d, decoupled_idx=%d, temp3_coupled=%f, temp3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_SP2[coupled_idx].temp30, host_SP2_dem[decoupled_idx].temp30);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].drho != host_P3_dem[decoupled_idx].drho){
				printf("[DEBUG] DEM mismatched drho value coupled_idx=%d, decoupled_idx=%d, drho_coupled=%f, drho_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].drho, host_P3_dem[decoupled_idx].drho);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dconcn != host_P3_dem[decoupled_idx].dconcn){
				printf("[DEBUG] DEM mismatched dconcn value coupled_idx=%d, decoupled_idx=%d, dconcn_coupled=%f, dconcn_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dconcn, host_P3_dem[decoupled_idx].dconcn);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].denthalpy != host_P3_dem[decoupled_idx].denthalpy){
				printf("[DEBUG] DEM mismatched denthalpy value coupled_idx=%d, decoupled_idx=%d, denthalpy_coupled=%f, denthalpy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].denthalpy, host_P3_dem[decoupled_idx].denthalpy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].drad != host_P3_dem[decoupled_idx].drad){
				printf("[DEBUG] DEM mismatched drad value coupled_idx=%d, decoupled_idx=%d, drad_coupled=%f, drad_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].drad, host_P3_dem[decoupled_idx].drad);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp != host_P3_dem[decoupled_idx].dtemp){
				printf("[DEBUG] DEM mismatched dtemp value coupled_idx=%d, decoupled_idx=%d, dtemp_coupled=%f, dtemp_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp, host_P3_dem[decoupled_idx].dtemp);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp1 != host_P3_dem[decoupled_idx].dtemp1){
				printf("[DEBUG] DEM mismatched dtemp1 value coupled_idx=%d, decoupled_idx=%d, dtemp1_coupled=%f, dtemp1_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp1, host_P3_dem[decoupled_idx].dtemp1);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp2 != host_P3_dem[decoupled_idx].dtemp2){
				printf("[DEBUG] DEM mismatched dtemp2 value coupled_idx=%d, decoupled_idx=%d, dtemp2_coupled=%f, dtemp2_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp2, host_P3_dem[decoupled_idx].dtemp2);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dtemp3 != host_P3_dem[decoupled_idx].dtemp3){
				printf("[DEBUG] DEM mismatched dtemp3 value coupled_idx=%d, decoupled_idx=%d, dtemp3_coupled=%f, dtemp3_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dtemp3, host_P3_dem[decoupled_idx].dtemp3);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].torqx != host_P3_dem[decoupled_idx].torqx){
				printf("[DEBUG] DEM mismatched torqx value coupled_idx=%d, decoupled_idx=%d, torqx_coupled=%f, torqx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].torqx, host_P3_dem[decoupled_idx].torqx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].torqy != host_P3_dem[decoupled_idx].torqy){
				printf("[DEBUG] DEM mismatched torqy value coupled_idx=%d, decoupled_idx=%d, torqy_coupled=%f, torqy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].torqy, host_P3_dem[decoupled_idx].torqy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].torqz != host_P3_dem[decoupled_idx].torqz){
				printf("[DEBUG] DEM mismatched torqz value coupled_idx=%d, decoupled_idx=%d, torqz_coupled=%f, torqz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].torqz, host_P3_dem[decoupled_idx].torqz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotalx != host_P3_dem[decoupled_idx].ftotalx){
				printf("[DEBUG] DEM mismatched ftotalx value coupled_idx=%d, decoupled_idx=%d, ftotalx_coupled=%f, ftotalx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotalx, host_P3_dem[decoupled_idx].ftotalx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotaly != host_P3_dem[decoupled_idx].ftotaly){
				printf("[DEBUG] DEM mismatched ftotaly value coupled_idx=%d, decoupled_idx=%d, ftotaly_coupled=%f, ftotaly_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotaly, host_P3_dem[decoupled_idx].ftotaly);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotalz != host_P3_dem[decoupled_idx].ftotalz){
				printf("[DEBUG] DEM mismatched ftotalz value coupled_idx=%d, decoupled_idx=%d, ftotalz_coupled=%f, ftotalz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotalz, host_P3_dem[decoupled_idx].ftotalz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ftotal != host_P3_dem[decoupled_idx].ftotal){
				printf("[DEBUG] DEM mismatched ftotal value coupled_idx=%d, decoupled_idx=%d, ftotal_coupled=%f, ftotal_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ftotal, host_P3_dem[decoupled_idx].ftotal);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fpx != host_P3_dem[decoupled_idx].fpx){
				printf("[DEBUG] DEM mismatched fpx value coupled_idx=%d, decoupled_idx=%d, fpx_coupled=%f, fpx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fpx, host_P3_dem[decoupled_idx].fpx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fpy != host_P3_dem[decoupled_idx].fpy){
				printf("[DEBUG] DEM mismatched fpy value coupled_idx=%d, decoupled_idx=%d, fpy_coupled=%f, fpy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fpy, host_P3_dem[decoupled_idx].fpy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fpz != host_P3_dem[decoupled_idx].fpz){
				printf("[DEBUG] DEM mismatched fpz value coupled_idx=%d, decoupled_idx=%d, fpz_coupled=%f, fpz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fpz, host_P3_dem[decoupled_idx].fpz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].vis_t != host_P3_dem[decoupled_idx].vis_t){
				printf("[DEBUG] DEM mismatched vis_t value coupled_idx=%d, decoupled_idx=%d, vis_t_coupled=%f, vis_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].vis_t, host_P3_dem[decoupled_idx].vis_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Sxx != host_P3_dem[decoupled_idx].Sxx){
				printf("[DEBUG] DEM mismatched Sxx value coupled_idx=%d, decoupled_idx=%d, Sxx_coupled=%f, Sxx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Sxx, host_P3_dem[decoupled_idx].Sxx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Sxy != host_P3_dem[decoupled_idx].Sxy){
				printf("[DEBUG] DEM mismatched Sxy value coupled_idx=%d, decoupled_idx=%d, Sxy_coupled=%f, Sxy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Sxy, host_P3_dem[decoupled_idx].Sxy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Sxz != host_P3_dem[decoupled_idx].Sxz){
				printf("[DEBUG] DEM mismatched Sxz value coupled_idx=%d, decoupled_idx=%d, Sxz_coupled=%f, Sxz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Sxz, host_P3_dem[decoupled_idx].Sxz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Syy != host_P3_dem[decoupled_idx].Syy){
				printf("[DEBUG] DEM mismatched Syy value coupled_idx=%d, decoupled_idx=%d, Syy_coupled=%f, Syy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Syy, host_P3_dem[decoupled_idx].Syy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Syz != host_P3_dem[decoupled_idx].Syz){
				printf("[DEBUG] DEM mismatched Syz value coupled_idx=%d, decoupled_idx=%d, Syz_coupled=%f, Syz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Syz, host_P3_dem[decoupled_idx].Syz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].Szz != host_P3_dem[decoupled_idx].Szz){
				printf("[DEBUG] DEM mismatched Szz value coupled_idx=%d, decoupled_idx=%d, Szz_coupled=%f, Szz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].Szz, host_P3_dem[decoupled_idx].Szz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].dk_turb != host_P3_dem[decoupled_idx].dk_turb){
				printf("[DEBUG] DEM mismatched dk_turb value coupled_idx=%d, decoupled_idx=%d, dk_turb_coupled=%f, dk_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].dk_turb, host_P3_dem[decoupled_idx].dk_turb);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].de_turb != host_P3_dem[decoupled_idx].de_turb){
				printf("[DEBUG] DEM mismatched de_turb value coupled_idx=%d, decoupled_idx=%d, de_turb_coupled=%f, de_turb_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].de_turb, host_P3_dem[decoupled_idx].de_turb);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].lbl_surf != host_P3_dem[decoupled_idx].lbl_surf){
				printf("[DEBUG] DEM mismatched lbl_surf value coupled_idx=%d, decoupled_idx=%d, lbl_surf_coupled=%f, lbl_surf_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].lbl_surf, host_P3_dem[decoupled_idx].lbl_surf);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].cc != host_P3_dem[decoupled_idx].cc){
				printf("[DEBUG] DEM mismatched cc value coupled_idx=%d, decoupled_idx=%d, cc_coupled=%f, cc_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].cc, host_P3_dem[decoupled_idx].cc);
				mismatch_count++;
			}
			for(int ci=0; ci<Correction_Matrix_Size; ci++){
				for(int cj=0; cj<Correction_Matrix_Size; cj++){
					if(host_P3[coupled_idx].Cm[ci][cj] != host_P3_dem[decoupled_idx].Cm[ci][cj]){
						printf("[DEBUG] DEM mismatched Cm[%d][%d] value coupled_idx=%d, decoupled_idx=%d, Cm_coupled=%f, Cm_decoupled=%f\n",
							ci, cj, coupled_idx, decoupled_idx, host_P3[coupled_idx].Cm[ci][cj], host_P3_dem[decoupled_idx].Cm[ci][cj]);
						mismatch_count++;
					}
				}
			}
			if(host_P3[coupled_idx].nx != host_P3_dem[decoupled_idx].nx){
				printf("[DEBUG] DEM mismatched nx value coupled_idx=%d, decoupled_idx=%d, nx_coupled=%f, nx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx, host_P3_dem[decoupled_idx].nx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny != host_P3_dem[decoupled_idx].ny){
				printf("[DEBUG] DEM mismatched ny value coupled_idx=%d, decoupled_idx=%d, ny_coupled=%f, ny_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny, host_P3_dem[decoupled_idx].ny);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz != host_P3_dem[decoupled_idx].nz){
				printf("[DEBUG] DEM mismatched nz value coupled_idx=%d, decoupled_idx=%d, nz_coupled=%f, nz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz, host_P3_dem[decoupled_idx].nz);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_u != host_P3_dem[decoupled_idx].nx_u){
				printf("[DEBUG] DEM mismatched nx_u value coupled_idx=%d, decoupled_idx=%d, nx_u_coupled=%f, nx_u_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_u, host_P3_dem[decoupled_idx].nx_u);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_u != host_P3_dem[decoupled_idx].ny_u){
				printf("[DEBUG] DEM mismatched ny_u value coupled_idx=%d, decoupled_idx=%d, ny_u_coupled=%f, ny_u_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_u, host_P3_dem[decoupled_idx].ny_u);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_u != host_P3_dem[decoupled_idx].nz_u){
				printf("[DEBUG] DEM mismatched nz_u value coupled_idx=%d, decoupled_idx=%d, nz_u_coupled=%f, nz_u_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_u, host_P3_dem[decoupled_idx].nz_u);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_w != host_P3_dem[decoupled_idx].nx_w){
				printf("[DEBUG] DEM mismatched nx_w value coupled_idx=%d, decoupled_idx=%d, nx_w_coupled=%f, nx_w_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_w, host_P3_dem[decoupled_idx].nx_w);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_w != host_P3_dem[decoupled_idx].ny_w){
				printf("[DEBUG] DEM mismatched ny_w value coupled_idx=%d, decoupled_idx=%d, ny_w_coupled=%f, ny_w_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_w, host_P3_dem[decoupled_idx].ny_w);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_w != host_P3_dem[decoupled_idx].nz_w){
				printf("[DEBUG] DEM mismatched nz_w value coupled_idx=%d, decoupled_idx=%d, nz_w_coupled=%f, nz_w_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_w, host_P3_dem[decoupled_idx].nz_w);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_t != host_P3_dem[decoupled_idx].nx_t){
				printf("[DEBUG] DEM mismatched nx_t value coupled_idx=%d, decoupled_idx=%d, nx_t_coupled=%f, nx_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_t, host_P3_dem[decoupled_idx].nx_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_t != host_P3_dem[decoupled_idx].ny_t){
				printf("[DEBUG] DEM mismatched ny_t value coupled_idx=%d, decoupled_idx=%d, ny_t_coupled=%f, ny_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_t, host_P3_dem[decoupled_idx].ny_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_t != host_P3_dem[decoupled_idx].nz_t){
				printf("[DEBUG] DEM mismatched nz_t value coupled_idx=%d, decoupled_idx=%d, nz_t_coupled=%f, nz_t_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_t, host_P3_dem[decoupled_idx].nz_t);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_tl != host_P3_dem[decoupled_idx].nx_tl){
				printf("[DEBUG] DEM mismatched nx_tl value coupled_idx=%d, decoupled_idx=%d, nx_tl_coupled=%f, nx_tl_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_tl, host_P3_dem[decoupled_idx].nx_tl);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_tl != host_P3_dem[decoupled_idx].ny_tl){
				printf("[DEBUG] DEM mismatched ny_tl value coupled_idx=%d, decoupled_idx=%d, ny_tl_coupled=%f, ny_tl_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_tl, host_P3_dem[decoupled_idx].ny_tl);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_tl != host_P3_dem[decoupled_idx].nz_tl){
				printf("[DEBUG] DEM mismatched nz_tl value coupled_idx=%d, decoupled_idx=%d, nz_tl_coupled=%f, nz_tl_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_tl, host_P3_dem[decoupled_idx].nz_tl);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nx_c != host_P3_dem[decoupled_idx].nx_c){
				printf("[DEBUG] DEM mismatched nx_c value coupled_idx=%d, decoupled_idx=%d, nx_c_coupled=%f, nx_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nx_c, host_P3_dem[decoupled_idx].nx_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].ny_c != host_P3_dem[decoupled_idx].ny_c){
				printf("[DEBUG] DEM mismatched ny_c value coupled_idx=%d, decoupled_idx=%d, ny_c_coupled=%f, ny_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].ny_c, host_P3_dem[decoupled_idx].ny_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nz_c != host_P3_dem[decoupled_idx].nz_c){
				printf("[DEBUG] DEM mismatched nz_c value coupled_idx=%d, decoupled_idx=%d, nz_c_coupled=%f, nz_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nz_c, host_P3_dem[decoupled_idx].nz_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nmag != host_P3_dem[decoupled_idx].nmag){
				printf("[DEBUG] DEM mismatched nmag value coupled_idx=%d, decoupled_idx=%d, nmag_coupled=%f, nmag_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nmag, host_P3_dem[decoupled_idx].nmag);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].nmag_c != host_P3_dem[decoupled_idx].nmag_c){
				printf("[DEBUG] DEM mismatched nmag_c value coupled_idx=%d, decoupled_idx=%d, nmag_c_coupled=%f, nmag_c_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].nmag_c, host_P3_dem[decoupled_idx].nmag_c);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].curv != host_P3_dem[decoupled_idx].curv){
				printf("[DEBUG] DEM mismatched curv value coupled_idx=%d, decoupled_idx=%d, curv_coupled=%f, curv_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].curv, host_P3_dem[decoupled_idx].curv);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fsx != host_P3_dem[decoupled_idx].fsx){
				printf("[DEBUG] DEM mismatched fsx value coupled_idx=%d, decoupled_idx=%d, fsx_coupled=%f, fsx_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fsx, host_P3_dem[decoupled_idx].fsx);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fsy != host_P3_dem[decoupled_idx].fsy){
				printf("[DEBUG] DEM mismatched fsy value coupled_idx=%d, decoupled_idx=%d, fsy_coupled=%f, fsy_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fsy, host_P3_dem[decoupled_idx].fsy);
				mismatch_count++;
			}
			if(host_P3[coupled_idx].fsz != host_P3_dem[decoupled_idx].fsz){
				printf("[DEBUG] DEM mismatched fsz value coupled_idx=%d, decoupled_idx=%d, fsz_coupled=%f, fsz_decoupled=%f\n",
					coupled_idx, decoupled_idx, host_P3[coupled_idx].fsz, host_P3_dem[decoupled_idx].fsz);
				mismatch_count++;
			}
			decoupled_idx++;
			if (mismatch_count > 0) {
				printf("[ERROR] SPH mismatch detected. Exiting.\n");
				exit(EXIT_FAILURE);
			}
		}
		coupled_idx++;
	}
	printf("[DEBUG] DEM Total mismatch count: %d\n", mismatch_count);
	free(host_SP1);
	free(host_SP1_dem);
}


void Check_cell_index_sph(
    int_t *g_idx_sph, int_t *p_idx_sph, int_t *g_idx_sph_in, int_t *p_idx_sph_in, int_t *g_str_sph, int_t *g_end_sph,
    int_t *g_idx_sph_dem, int_t *p_idx_sph_dem, int_t *g_idx_sph_in_dem, int_t *p_idx_sph_in_dem, int_t *g_str_sph_dem, int_t *g_end_sph_dem)
{
    int_t *host_g_idx_sph = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_p_idx_sph = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_g_idx_sph_in = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_p_idx_sph_in = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_g_str_sph = (int_t*)malloc(sizeof(int_t)*num_cells);
    int_t *host_g_end_sph = (int_t*)malloc(sizeof(int_t)*num_cells);

    int_t *host_g_idx_sph_dem = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_p_idx_sph_dem = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_g_idx_sph_in_dem = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_p_idx_sph_in_dem = (int_t*)malloc(sizeof(int_t)*num_part2_sph);
    int_t *host_g_str_sph_dem = (int_t*)malloc(sizeof(int_t)*num_cells);
    int_t *host_g_end_sph_dem = (int_t*)malloc(sizeof(int_t)*num_cells);

    CUDA_CHECK(cudaMemcpy(host_g_idx_sph, g_idx_sph, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_sph, p_idx_sph, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_idx_sph_in, g_idx_sph_in, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_sph_in, p_idx_sph_in, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_str_sph, g_str_sph, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_end_sph, g_end_sph, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(host_g_idx_sph_dem, g_idx_sph_dem, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_sph_dem, p_idx_sph_dem, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_idx_sph_in_dem, g_idx_sph_in_dem, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_sph_in_dem, p_idx_sph_in_dem, sizeof(int_t)*num_part2_sph, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_str_sph_dem, g_str_sph_dem, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_end_sph_dem, g_end_sph_dem, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    int mismatch_count = 0;
    // Check arrays with num_part2_sph size
    for (int i = 0; i < num_part2_sph; i++) {
        if (host_g_idx_sph[i] != host_g_idx_sph_dem[i]) {
            printf("[ERROR] Mismatch in g_idx_sph at index %d: SPH=%d, DEM=%d\n", i, host_g_idx_sph[i], host_g_idx_sph_dem[i]);
            mismatch_count++;
        }
        if (host_p_idx_sph[i] != host_p_idx_sph_dem[i]) {
            printf("[ERROR] Mismatch in p_idx_sph at index %d: SPH=%d, DEM=%d\n", i, host_p_idx_sph[i], host_p_idx_sph_dem[i]);
            mismatch_count++;
        }
        if (host_g_idx_sph_in[i] != host_g_idx_sph_in_dem[i]) {
            printf("[ERROR] Mismatch in g_idx_sph_in at index %d: SPH=%d, DEM=%d\n", i, host_g_idx_sph_in[i], host_g_idx_sph_in_dem[i]);
            mismatch_count++;
        }
        if (host_p_idx_sph_in[i] != host_p_idx_sph_in_dem[i]) {
            printf("[ERROR] Mismatch in p_idx_sph_in at index %d: SPH=%d, DEM=%d\n", i, host_p_idx_sph_in[i], host_p_idx_sph_in_dem[i]);
            mismatch_count++;
        }
        if (mismatch_count > 0) {
            printf("[ERROR] SPH/DEM integer array mismatch detected. Exiting.\n");
            exit(EXIT_FAILURE);
        }
    }
    // Check arrays with num_cell size
    for (int i = 0; i < num_cells; i++) {
        if (host_g_str_sph[i] != host_g_str_sph_dem[i]) {
            printf("[ERROR] Mismatch in g_str_sph at index %d: SPH=%d, DEM=%d\n", i, host_g_str_sph[i], host_g_str_sph_dem[i]);
            mismatch_count++;
        }
        if (host_g_end_sph[i] != host_g_end_sph_dem[i]) {
            printf("[ERROR] Mismatch in g_end_sph at index %d: SPH=%d, DEM=%d\n", i, host_g_end_sph[i], host_g_end_sph_dem[i]);
            mismatch_count++;
        }
        if (mismatch_count > 0) {
            printf("[ERROR] SPH/DEM integer array mismatch detected. Exiting.\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("[DEBUG] SPH/DEM integer arrays are identical.\n");

    free(host_g_idx_sph);
    free(host_p_idx_sph);
    free(host_g_idx_sph_in);
    free(host_p_idx_sph_in);
    free(host_g_str_sph);
    free(host_g_end_sph);
    free(host_g_idx_sph_dem);
    free(host_p_idx_sph_dem);
    free(host_g_idx_sph_in_dem);
    free(host_p_idx_sph_in_dem);
    free(host_g_str_sph_dem);
    free(host_g_end_sph_dem);
}

void Check_cell_index_dem(
    int_t *g_idx_dem, int_t *p_idx_dem, int_t *g_idx_dem_in, int_t *p_idx_dem_in, int_t *g_str_dem, int_t *g_end_dem,
    int_t *g_idx_dem_dem, int_t *p_idx_dem_dem, int_t *g_idx_dem_in_dem, int_t *p_idx_dem_in_dem, int_t *g_str_dem_dem, int_t *g_end_dem_dem)
{
    int_t *host_g_idx_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_p_idx_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_g_idx_dem_in = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_p_idx_dem_in = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_g_str_dem = (int_t*)malloc(sizeof(int_t)*num_cells);
    int_t *host_g_end_dem = (int_t*)malloc(sizeof(int_t)*num_cells);

    int_t *host_g_idx_dem_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_p_idx_dem_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_g_idx_dem_in_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_p_idx_dem_in_dem = (int_t*)malloc(sizeof(int_t)*num_part2_dem);
    int_t *host_g_str_dem_dem = (int_t*)malloc(sizeof(int_t)*num_cells);
    int_t *host_g_end_dem_dem = (int_t*)malloc(sizeof(int_t)*num_cells);

    CUDA_CHECK(cudaMemcpy(host_g_idx_dem, g_idx_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_dem, p_idx_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_idx_dem_in, g_idx_dem_in, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_dem_in, p_idx_dem_in, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_str_dem, g_str_dem, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_end_dem, g_end_dem, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(host_g_idx_dem_dem, g_idx_dem_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_dem_dem, p_idx_dem_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_idx_dem_in_dem, g_idx_dem_in_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_p_idx_dem_in_dem, p_idx_dem_in_dem, sizeof(int_t)*num_part2_dem, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_str_dem_dem, g_str_dem_dem, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_g_end_dem_dem, g_end_dem_dem, sizeof(int_t)*num_cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    int mismatch_count = 0;
    // Check arrays with num_part2_dem size
    for (int i = 0; i < num_part2_dem; i++) {
        if (host_g_idx_dem[i] != host_g_idx_dem_dem[i]) {
            printf("[ERROR] Mismatch in g_idx_dem at index %d: DEM=%d, DEM=%d\n", i, host_g_idx_dem[i], host_g_idx_dem_dem[i]);
            mismatch_count++;
        }
        if (host_p_idx_dem[i] != host_p_idx_dem_dem[i]) {
            printf("[ERROR] Mismatch in p_idx_dem at index %d: DEM=%d, DEM=%d\n", i, host_p_idx_dem[i], host_p_idx_dem_dem[i]);
            mismatch_count++;
        }
        if (host_g_idx_dem_in[i] != host_g_idx_dem_in_dem[i]) {
            printf("[ERROR] Mismatch in g_idx_dem_in at index %d: DEM=%d, DEM=%d\n", i, host_g_idx_dem_in[i], host_g_idx_dem_in_dem[i]);
            mismatch_count++;
        }
        if (host_p_idx_dem_in[i] != host_p_idx_dem_in_dem[i]) {
            printf("[ERROR] Mismatch in p_idx_dem_in at index %d: DEM=%d, DEM=%d\n", i, host_p_idx_dem_in[i], host_p_idx_dem_in_dem[i]);
            mismatch_count++;
        }
        if (mismatch_count > 0) {
            printf("[ERROR] SPH/DEM integer array mismatch detected. Exiting.\n");
            exit(EXIT_FAILURE);
        }

    }
    // Check arrays with num_cell size
    for (int i = 0; i < num_cells; i++) {
        if (host_g_str_dem[i] != host_g_str_dem_dem[i]) {
            printf("[ERROR] Mismatch in g_str_dem at index %d: DEM=%d, DEM=%d\n", i, host_g_str_dem[i], host_g_str_dem_dem[i]);
            mismatch_count++;
        }
        if (host_g_end_dem[i] != host_g_end_dem_dem[i]) {
            printf("[ERROR] Mismatch in g_end_dem at index %d: DEM=%d, DEM=%d\n", i, host_g_end_dem[i], host_g_end_dem_dem[i]);
            mismatch_count++;
        }
        if (mismatch_count > 0) {
            printf("[ERROR] SPH/DEM integer array mismatch detected. Exiting.\n");
            exit(EXIT_FAILURE);
        }
    }
    if (mismatch_count > 0) {
        printf("[ERROR] DEM integer array mismatch detected. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    printf("[DEBUG] DEM integer arrays are identical.\n");

    free(host_g_idx_dem);
    free(host_p_idx_dem);
    free(host_g_idx_dem_in);
    free(host_p_idx_dem_in);
    free(host_g_str_dem);
    free(host_g_end_dem);
    free(host_g_idx_dem_dem);
    free(host_p_idx_dem_dem);
    free(host_g_idx_dem_in_dem);
    free(host_p_idx_dem_in_dem);
    free(host_g_str_dem_dem);
    free(host_g_end_dem_dem);
}