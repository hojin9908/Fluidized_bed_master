double cclssw[256],cclgsw[256];  // COtimers
////////////////////////////////////////////////////////////////////////
double setsw(int seq)
{
	struct timespec AA;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&AA);
	cclssw[seq]=(double)AA.tv_sec+(double)AA.tv_nsec/1000000000.0;
	return(cclssw[seq]);
}
////////////////////////////////////////////////////////////////////////
double getsw(int seq)
{
	struct timespec AA;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&AA);
	cclgsw[seq]=(double)AA.tv_sec+(double)AA.tv_nsec/1000000000.0-cclssw[seq];
	return(cclgsw[seq]);
}
////////////////////////////////////////////////////////////////////////
__global__ void kernel_copy_max(part1*P1,part3*P3,Real*mrho,Real*mft,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		mrho[i]=0;
		mft[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
	mrho[i]=P1[i].rho;
	mft[i]=P3[i].ftotal;
}

__global__ void kernel_copy_max_sph(part1*P1,part3*P3,Real*mrho,Real*mft,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		mrho[i]=0;
		mft[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
	mrho[i]=P1[i].rho;
	mft[i]=P3[i].ftotal;
}

__global__ void kernel_copy_max_dem(part1*P1,part3*P3,Real*mrho,Real*mft,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		mrho[i]=0;
		mft[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
	mrho[i]=P1[i].rho;
	mft[i]=P3[i].ftotal;
}


float FloatSwap( float f )
{
   union
   {
      float f;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.f=f;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
	 /*
   dat2.b[0]=dat1.b[7];
   dat2.b[1]=dat1.b[6];
   dat2.b[2]=dat1.b[5];
   dat2.b[3]=dat1.b[4];
   dat2.b[4]=dat1.b[3];
   dat2.b[5]=dat1.b[2];
   dat2.b[6]=dat1.b[1];
   dat2.b[7]=dat1.b[0];
	 //*/

   return dat2.f;
}
////////////////////////////////////////////////////////////////////////
int IntSwap( int d )
{
   union
   {
      int d;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.d=d;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
	 /*
   dat2.b[0]=dat1.b[7];
   dat2.b[1]=dat1.b[6];
   dat2.b[2]=dat1.b[5];
   dat2.b[3]=dat1.b[4];
   dat2.b[4]=dat1.b[3];
   dat2.b[5]=dat1.b[2];
   dat2.b[6]=dat1.b[1];
   dat2.b[7]=dat1.b[0];
	 //*/
   return dat2.d;
}



////////////////////////////////////////////////////////////////////////
void save_plot_fluid_vtk_bin(part1*P1, part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if((P1[i].i_type==1)||(P1[i].i_type==2)) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if((P1[i].i_type==1)||(P1[i].i_type==2)){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t3\n");


	fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if((P1[i].i_type==1)||(P1[i].i_type==2)){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"ptype\t1\t%d\tint\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if((P1[i].i_type==1)||(P1[i].i_type==2)){
			valt=IntSwap(P1[i].p_type);
			fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pres\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		if((P1[i].i_type==1)||(P1[i].i_type==2)){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
void save_plot_fluid_vtk_bin_fluid(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==1) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t12\n");



	fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].DEMpor);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressuregrad_x\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].pgf_x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressuregrad_y\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].pgf_y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressuregrad_z\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].pgf_z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	fprintf(outFile_vtk,"ux\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].ux);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"drho\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].drho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }



	fprintf(outFile_vtk,"uy\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].uy);	
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"dtemp\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P3[i].dtemp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

			fprintf(outFile_vtk,"buffer_type\t1\t%d\tint\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			valt=IntSwap(P1[i].buffer_type);
			fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
		}
	}

	// 		fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		valt=IntSwap(P1[i].i_type);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"temperature\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P1[i].rho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	//fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	//for(i=0;i<nop;i++){
	//	//if(P1[i].x>0){
	//	if(P1[i].p_type==1){
	//		val=FloatSwap(P1[i].DEMpor);
	//		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	//	}
	//}

	//fprintf(outFile_vtk,"pgf_z\t1\t%d\tfloat\n",Nparticle);
	//for(i=0;i<nop;i++){
	//	//if(P1[i].x>0){
	//	if(P1[i].p_type==1){
	//		val=FloatSwap(P1[i].pgf_z);
	//		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	//	}
	//}

	
	// fprintf(outFile_vtk,"du_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].ftotalz);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"drho_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].drho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	// fprintf(outFile_vtk,"pgf_z\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P1[i].pgf_z);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }








	fclose(outFile_vtk);
}


void save_plot_fluid_vtk_bin_fluid_sph(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2_sph;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==1) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/sphfluid_decouple%d_dt%.0E_%dstp.vtk",decouple_stride,dt,count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t9\n");



	fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].DEMpor);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"F_total\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P1[i].Fdz_df);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"conv_rate\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].Q_sdf);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"drho\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].drho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }



	fprintf(outFile_vtk,"cond_rate\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].Q_f);	
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"dtemp\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P3[i].dtemp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

			fprintf(outFile_vtk,"buffer_type\t1\t%d\tint\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			valt=IntSwap(P1[i].buffer_type);
			fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
		}
	}

	// 		fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		valt=IntSwap(P1[i].i_type);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"temperature\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P1[i].rho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	//fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	//for(i=0;i<nop;i++){
	//	//if(P1[i].x>0){
	//	if(P1[i].p_type==1){
	//		val=FloatSwap(P1[i].DEMpor);
	//		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	//	}
	//}

	//fprintf(outFile_vtk,"pgf_z\t1\t%d\tfloat\n",Nparticle);
	//for(i=0;i<nop;i++){
	//	//if(P1[i].x>0){
	//	if(P1[i].p_type==1){
	//		val=FloatSwap(P1[i].pgf_z);
	//		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	//	}
	//}

	
	// fprintf(outFile_vtk,"du_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].ftotalz);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"drho_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].drho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==1){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	// fprintf(outFile_vtk,"pgf_z\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P1[i].pgf_z);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	fclose(outFile_vtk);
}

void save_plot_fluid_vtk_bin_air(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==3 ) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/air_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==3){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t10\n");


	fprintf(outFile_vtk,"rho_ref\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==3){
			val=FloatSwap(P2[i].rho_ref);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	fprintf(outFile_vtk,"buffer_type\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==3){
			val=FloatSwap(P1[i].buffer_type);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==3){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"temperature\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==3){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	// fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if((P1[i].p_type==3) | (P1[i].p_type==-3)){
	// 		val=FloatSwap(P1[i].rho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
			if(P1[i].p_type==3){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
			if(P1[i].p_type==3){
			val=FloatSwap(P1[i].DEMpor);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
			if(P1[i].p_type==3){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"drho_dt\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
			if(P1[i].p_type==3){
			val=FloatSwap(P3[i].drho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"dtemp_dt\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
			if(P1[i].p_type==3){
			val=FloatSwap(P3[i].dtemp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	// fprintf(outFile_vtk,"pgf_z\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 		if((P1[i].p_type==3) | (P1[i].p_type==-3)){
	// 		val=FloatSwap(P1[i].pgf_z);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	fprintf(outFile_vtk,"ptype\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==3){
			val=FloatSwap(P1[i].p_type);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}




	fclose(outFile_vtk);
}


void save_plot_fluid_vtk_bin_solid(part1*P1,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].dem_idx>0) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/DEM_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t10\n");




	fprintf(outFile_vtk,"dem_idx\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].dem_idx);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"mag_uijf\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].test6);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"drag_r\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){	
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].test7);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}







	

	fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].DEMpor);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	



	fprintf(outFile_vtk,"Q_sd\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].Q_sd);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	




	fprintf(outFile_vtk,"ftotal_z\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P3[i].ftotalz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	// fprintf(outFile_vtk,"ftotalx\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].ftotalx);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// } 



	fprintf(outFile_vtk,"temperature_surface\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}




	// fprintf(outFile_vtk,"dtemp1_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].dtemp1);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }





	fprintf(outFile_vtk,"rad\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].rad);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	fprintf(outFile_vtk,"dtemp_dt\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P3[i].dtemp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	// fprintf(outFile_vtk,"mag_uij\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].pres);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"ftotaly\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].ftotaly);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// 	fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		valt=IntSwap(P1[i].i_type);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"demidx\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		valt=IntSwap(P1[i].dem_idx);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"flt_s\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].flt_s);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"flt_sd\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].flt_sd);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"fdz_b\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].Fdz_b);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }


	// fprintf(outFile_vtk,"curvature\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].curv);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"detection\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].lbl_surf);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"surface_tension\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].fsx);
	// 		val2=FloatSwap(P3[i].fsy);
	// 		val3=FloatSwap(P3[i].fsz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector_c\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx_c);
	// 		val2=FloatSwap(P3[i].ny_c);
	// 		val3=FloatSwap(P3[i].nz_c);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx);
	// 		val2=FloatSwap(P3[i].ny);
	// 		val3=FloatSwap(P3[i].nz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fclose(outFile_vtk);
}


void save_plot_fluid_vtk_bin_solid_dem(part1*P1,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2_dem;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].dem_idx>0) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/demDEM_decouple%d_dt%.0E_%dstp.vtk",decouple_stride,dt,count);
	// sprintf(FileName_vtk,"./plotdata/demDEM_decouple%d_dt%.2E_%dstp.vtk",decouple_stride,dt,count);
	// sprintf(FileName_vtk,"./plotdata/demDEM_decouple%d_%dstp.vtk",decouple_stride,count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t10\n");




	fprintf(outFile_vtk,"itype\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].i_type);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"mag_uijf\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].test6);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"Euler\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){	
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].eliz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}







	

	fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].DEMpor);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	



	fprintf(outFile_vtk,"Q_sd\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].Q_sd);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	




	fprintf(outFile_vtk,"ftotal_z\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P3[i].ftotalz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	// fprintf(outFile_vtk,"ftotalx\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].ftotalx);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// } 



	fprintf(outFile_vtk,"temperature_surface\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}




	// fprintf(outFile_vtk,"dtemp1_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].dtemp1);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }





	fprintf(outFile_vtk,"rad\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].rad);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	fprintf(outFile_vtk,"dtemp_dt\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P3[i].dtemp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	// fprintf(outFile_vtk,"mag_uij\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].pres);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"ftotaly\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].ftotaly);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// 	fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		valt=IntSwap(P1[i].i_type);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"demidx\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		valt=IntSwap(P1[i].dem_idx);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"flt_s\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].flt_s);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"flt_sd\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].flt_sd);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"fdz_b\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].Fdz_b);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }


	// fprintf(outFile_vtk,"curvature\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].curv);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"detection\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].lbl_surf);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"surface_tension\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].fsx);
	// 		val2=FloatSwap(P3[i].fsy);
	// 		val3=FloatSwap(P3[i].fsz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector_c\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx_c);
	// 		val2=FloatSwap(P3[i].ny_c);
	// 		val3=FloatSwap(P3[i].nz_c);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx);
	// 		val2=FloatSwap(P3[i].ny);
	// 		val3=FloatSwap(P3[i].nz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fclose(outFile_vtk);
}

void save_plot_fluid_vtk_bin_moving(part1*P1,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==MOVING) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/moving_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==MOVING){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t2\n");


	// fprintf(outFile_vtk,"p_type\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==MOVING){
	// 		valt=IntSwap(P1[i].p_type);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==MOVING){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==MOVING){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].curv);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"detection\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].lbl_surf);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"surface_tension\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].fsx);
	// 		val2=FloatSwap(P3[i].fsy);
	// 		val3=FloatSwap(P3[i].fsz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector_c\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx_c);
	// 		val2=FloatSwap(P3[i].ny_c);
	// 		val3=FloatSwap(P3[i].nz_c);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx);
	// 		val2=FloatSwap(P3[i].ny);
	// 		val3=FloatSwap(P3[i].nz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
void save_plot_fluid_vtk_bin_fluid2(part1*P1,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==2) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/fluid2_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==2){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t7\n");


	fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==2){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"pres\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==2){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"curvature\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==2){
			val=FloatSwap(P3[i].curv);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"detection\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==2){
			val=FloatSwap(P3[i].lbl_surf);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"surface_tension\t3\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==2){
			val1=FloatSwap(P3[i].fsx);
			val2=FloatSwap(P3[i].fsy);
			val3=FloatSwap(P3[i].fsz);
			fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
			fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
			fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
		}
	}

		fprintf(outFile_vtk,"normal_vector_c\t3\t%d\tfloat\n",Nparticle);
		for(i=0;i<nop;i++){
			//if(P1[i].x>0){
			if(P1[i].p_type==2){
				val1=FloatSwap(P3[i].nx_c);
				val2=FloatSwap(P3[i].ny_c);
				val3=FloatSwap(P3[i].nz_c);
				fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
				fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
				fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
			}
		}

		fprintf(outFile_vtk,"normal_vector\t3\t%d\tfloat\n",Nparticle);
		for(i=0;i<nop;i++){
			//if(P1[i].x>0){
			if(P1[i].p_type==2){
				val1=FloatSwap(P3[i].nx);
				val2=FloatSwap(P3[i].ny);
				val3=FloatSwap(P3[i].nz);
				fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
				fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
				fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
			}
		}

	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
void save_plot_fluid_vtk_bin_boundary(part1*P1)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==0) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/boundary_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t6\n");


	fprintf(outFile_vtk,"uy\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].uy);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"ux\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].ux);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"K\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==0){
	// 		val=FloatSwap(P1[i].test7);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"temperature\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	// fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==0){
	// 		val=FloatSwap(P1[i].uz);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"mass\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==0){
	// 		val=FloatSwap(P1[i].m);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"volume\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==0){
	// 		val=FloatSwap(P1[i].vol);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].pres);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"density\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
void save_plot_moving_vtk_bin(part1*P1)
{
	int_t i,nop;
	nop=num_part2;
	// int_t Nparticle=nop;									// number of fluid particles
	int_t Nparticle=0;									// number of fluid particles
	for(i=0;i<nop;i++) if((P1[i].p_type==9)) Nparticle++;
	// printf("%d test Particle %d\n",tid,Nparticle);

	//*
	float val;
	//int valt;


	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	// If the file already exists,its contents are discarded and create the new one.
	char FileName_vtk[256];
	// sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count);
	sprintf(FileName_vtk,"./plotdata/moving_%dstp.vtk",count);
	FILE*outFile_vtk;

	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);

	// fprintf(outFile_vtk,"%d\n",Nparticle);
	//Define SPH particles---------------------------------------------------------------
	// fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
	{
		if((P1[i].p_type==9)){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);
	fprintf(outFile_vtk,"FIELD FieldData\t2\n");

	fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		if(P1[i].p_type==9){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	fprintf(outFile_vtk,"uy\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		if(P1[i].p_type==9){
			val=FloatSwap(P1[i].uy);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	// for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
	// {
	// 	if(P1[i].i_type==1){
	// 		val=FloatSwap(P1[i].pres);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	// for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
	// {
	// 	if(P1[i].i_type==1){
	// 		val=FloatSwap(P1[i].rho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	fclose(outFile_vtk);
	//*/
}
////////////////////////////////////////////////////////////////////////
//S.H.Park
void save_restart(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part;
	//int_t Nparticle=nop;									// number of fluid particles

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName[256];
	sprintf(FileName,"./plotdata/restart.txt");
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile;
	outFile=fopen(FileName,"w");

	fprintf(outFile,"1 2 3 4 5 6 7 8 9 10 11 12 13 26\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)

	//Write data -------------------------------------------------------------------------
	for(i=0;i<nop;i++){
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].x,P1[i].y,P1[i].z);
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].ux,P1[i].uy,P1[i].uz);
			fprintf(outFile,"%e\t%d\t%e\t",P1[i].m,P1[i].p_type,P1[i].h);
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].temp,P1[i].pres,P1[i].rho);
			fprintf(outFile,"%f\t%f\n",P2[i].rho_ref,P1[i].flt_s);	//check f_total
			// fprintf(outFile,"%f\t%f\t%f\t",P2[i].rho_ref,P3[i].ftotal,P1[i].concn);	//check f_total
			// fprintf(outFile,"%f\t%f\t%d\t",P3[i].cc,P3[i].vis_t,P1[i].ct_boundary);
			// //fprintf(outFile,"%f\t%f\t%d\t%d\t",P3[i].cc,P3[i].vis_t,P1[i].ct_boundary,P3[i].hf_boundary);
			// fprintf(outFile,"%f\t%f\t%f\t%f\t%f\t%f\n",P3[i].lbl_surf,P3[i].drho,P3[i].denthalpy,P3[i].dconcn,P1[i].k_turb,P1[i].e_turb);
	}

	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
void save_vtk_bin_single(part1*P1)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if((P1[i].p_type==1)||(P1[i].p_type==2)) Nparticle++;
	printf("Number of Particles = %d\n\n",Nparticle);

	float val;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if((P1[i].p_type==1)||(P1[i].p_type==2)){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t%d\n",num_plot_data);

	for (int ccount=0;ccount<num_plot_data;ccount++)
	{
		char data_label[20];
		strcpy(data_label,plot_data[ccount]);

		// buffer_type
		if (!strncmp(data_label,"buffer_type",3)) {
			fprintf(outFile_vtk,"buffer_type\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					valt=IntSwap(P1[i].buffer_type);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		// i_type
		if (!strncmp(data_label,"i_type",3)) {
			fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					valt=IntSwap(P1[i].i_type);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		// p_type
		if (!strncmp(data_label,"p_type",3)) {
			fprintf(outFile_vtk,"p_type\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					valt=IntSwap(P1[i].p_type);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		// density
		if (!strncmp(data_label,"rho",3)) {
			fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].rho);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// ux
		if (!strncmp(data_label,"ux",2)) {
			fprintf(outFile_vtk,"ux\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].ux);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// uy
		if (!strncmp(data_label,"uy",3)) {
			fprintf(outFile_vtk,"uy\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].uy);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// uz
		if (!strncmp(data_label,"uz",3)) {
			fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].uz);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// pressure
		if (!strncmp(data_label,"pressure",3)) {
			fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].pres);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// temp
		if (!strncmp(data_label,"temp",3)) {
			fprintf(outFile_vtk,"temp\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].temp);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// flt_s
		if (!strncmp(data_label,"flt_s",3)) {
			fprintf(outFile_vtk,"flt_s\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].flt_s);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// concn
		if (!strncmp(data_label,"concn",3)) {
			fprintf(outFile_vtk,"concn\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if((P1[i].p_type==1)||(P1[i].p_type==2)){
					val=FloatSwap(P1[i].concn);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

	}



	fclose(outFile_vtk);
}
