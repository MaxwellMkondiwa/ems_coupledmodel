'''
EMA Workbench interface for NetLogo/SEAWAT coupled models

'''

import os
import glob
import shutil
import logging

import numpy as np

import pyNetLogo
import jpype


from ema_workbench.em_framework.model import FileModel, WorkingDirectoryModel
from ema_workbench.em_framework.model import Replicator, SingleReplication
from ema_workbench.util.ema_logging import warning, debug, info

import flopy.modflow as mf
import flopy.mt3d as mt3
import flopy.seawat as swt
import flopy.utils.binaryfile as bf

from agent_functions import PyWell, PySystem, PyGrid, createobjfromNetLogo,\
                            read_NetLogo_attriblist, write_NetLogo_attriblist, create_conc_list,\
                            create_LRCQ_list, update_runtime_objectlist, read_NetLogo_attrib,\
                            write_NetLogo_global, grid_interpolate, subsurface_use, boundaries

class ToyModel_MSI(FileModel):
    '''
    Model class for the iEMSs toy model
    '''
    
    command_format = 'set {0} {1}'

    def __init__(self, name, wd=None, model_file=None):
        '''
        Interface to the model
        
        :param working_directory: working_directory for the model. 
        :param name: name of the modelInterface. The name should contain only
                     alphanumerical characters. 
        '''
        
        # self.name = name
        # self.working_directory = wd

        super(ToyModel_MSI, self).__init__(name, wd=wd, model_file=model_file)
        # if working_directory:
        #     self.set_working_directory(working_directory)

    
    def model_init(self, policy, *args):
        '''
        Method called to initialize the model.
        
        :param policy: policy to be run.
        :param args: arguments to be used by model_init. This
                     gives users the ability to pass any additional 
                     arguments. 
        '''
        super(ToyModel_MSI, self).model_init(policy)   

        os.chdir(self.working_directory)

        logging.basicConfig(level=logging.INFO)
        self.dirs = [self.working_directory+r'/output']

        self.netlogo = pyNetLogo.NetLogoLink(self.GUI)
        logging.info('NetLogo started')
        self.netlogo.load_model(self.netlogo_filename)
        logging.info('Model opened')
    
    
    def run_experiment(self, experiment):
        '''
        Method for running an instantiated model structure. 
        
        This method should always be implemented.
        
        :param case: keyword arguments for running the model. The case is a 
                     dict with the names of the uncertainties as key, and
                     the values to which to set these uncertainties. 
        '''
             
         
        #NetLogo agent attributes to be passed to Python well objects
        #when new wells are created in NetLogo
        nl_read_sys_attribs = ['who','xcor','ycor']
        nl_read_well_attribs = ['who','xcor','ycor','IsCold','z0','FilterLength',
                                'T_inj','Q']
         
        #NetLogo agent attributes to be updated by the Python objects after each period
        nl_update_well_attribs = ['T_modflow','H_modflow']
        nl_update_globals = ['ztop','Laquifer']
        
        self.netlogo.command('setup')

        for key, value in experiment.items():
            if key in self.NetLogo_uncertainties:
                try:
                    self.netlogo.command(self.command_format.format(key, value))
                except jpype.JavaException as e:
                    warning('Variable {0} throws exception: {}'.format((key,
                                                                    str(e))))  
                logging.debug(self.netlogo.report(str(key)))
            if key in self.SEAWAT_uncertainties:
                setattr(self, key, value)

        #Set policy parameters if present
        if self.policy:
            for key, value in self.policy.items():
                if (key in self.NetLogo_uncertainties and key != 'name'):
                    self.netlogo.command(self.command_format.format(key, value))
                elif key in self.SEAWAT_uncertainties:
                    setattr(self, key, value)
            logging.info('Policy parameters set successfully')

        #Update NetLogo globals from input parameters
        for var in nl_update_globals:
            self.netlogo.command(self.command_format.format(var, getattr(self, var)))
        
        #Run the NetLogo setup routine, creating the agents
        #Create lists of Python objects based on the NetLogo agents
        self.netlogo.command('init-agents')
        sys_obj_list = update_runtime_objectlist(self.netlogo, [], nl_read_sys_attribs, breed='system', objclass=PySystem)
        well_obj_list, newgrid_flag = update_runtime_objectlist(self.netlogo, [], nl_read_well_attribs, breed='well', objclass=PyWell)

        #Assign values for uncertain NetLogo parameters
        logging.info('NetLogo parameters set successfully')

        #self.netlogo.command('init-agents')
        
        #Calculate geohydrological parameters linked to variable inputs
        rho_b = self.rho_solid * (1-self.PEFF)
        kT_b = self.kT_s * (1-self.PEFF) + self.kT_f * self.PEFF
        dmcoef = kT_b / (self.PEFF*self.rho_f*self.Cp_f) * 24 * 3600
        trpt = self.al * self.trp_mult
        trpv = trpt        

        #Initialize PyGrid object
        itype = mt3.Mt3dSsm.itype_dict()
        grid_obj = PyGrid()
        grid_obj.make_grid(well_obj_list, dmin=self.dmin, dmax=self.dmax, 
                           dz=self.dz, ztop=self.ztop, zbot=self.zbot, nstep=self.nstep, grid_extents=self.grid_extents)
          
        #Initial arrays for grid values (temperature, head) - for this case, assumes no groundwater flow
        #and uniform temperature
        grid_obj.ncol = len(grid_obj.XGR) - 1
        grid_obj.delr = np.diff(grid_obj.XGR)
        grid_obj.nrow = len(grid_obj.YGR) - 1 
        grid_obj.delc = -np.diff(grid_obj.YGR)

        grid_obj.top = self.ztop * np.ones([grid_obj.nrow, grid_obj.ncol])
        botm_range = np.arange(self.zbot, self.ztop, self.dz)[::-1]
        botm_2d = np.ones([grid_obj.nrow, grid_obj.ncol])
        grid_obj.botm = botm_2d*botm_range[:, None, None]
        grid_obj.nlay = len(botm_range)

        grid_obj.IBOUND, grid_obj.ICBUND = boundaries(grid_obj) #Create grid boundaries
        
        #Initial arrays for grid values (temperature, head)
        init_grid = np.ones((grid_obj.nlay, grid_obj.nrow, grid_obj.ncol))
        grid_obj.temp = 10.*init_grid
        
        grid_obj.HK = self.HK*init_grid
        grid_obj.VK = self.VK*init_grid

        #Set initial heads according to groundwater flow (based on mfLab Utrecht model)
        y_array = np.array([(grid_obj.YGR[:-1]-np.mean(grid_obj.YGR[:-1]))*self.PEFF*-self.gwflow_y/365/self.HK])
        y_tile = np.array([np.tile(y_array.T, (1, grid_obj.ncol))])
        x_array = (grid_obj.XGR[:-1]-np.mean(grid_obj.XGR[:-1]))*self.PEFF*-self.gwflow_x/365/self.HK
        y_tile += x_array
        grid_obj.head = np.tile(y_tile, (grid_obj.nlay, 1, 1))
        
        #Set times at which to read SEAWAT output for each simulation period
        timprs = np.array([self.perlen])
        nprs = len(timprs)
        logging.info('SEAWAT parameters set successfully')


        #Iterate the coupled model
        for period in range(self.run_length):

            #Set up the text output from NetLogo
            commands = []
            self.fns = {}
            for outcome in self.outcomes:
                #if outcome.time:
                name = outcome.name
                fn = r'{0}{3}{1}{2}'.format(self._working_directory,
                               name,
                               '.txt',
                               os.sep)
                self.fns[name] = fn
                fn = '"{}"'.format(fn)
                fn = fn.replace(os.sep, '/')
                
                if self.netlogo.report('is-agentset? {}'.format(name)):
                    #If name is name of an agentset, we
                    #assume that we should count the total number of agents
                    nc = r'{2} {0} {3} {4} {1}'.format(fn,
                                                       name,
                                                       'file-open',
                                                       'file-write',
                                                       'count')
                else:
                    #It is not an agentset, so assume that it is 
                    #a reporter / global variable
                    nc = r'{2} {0} {3} {1}'.format(fn,
                                                       name,
                                                       'file-open',
                                                       'file-write')
                commands.append(nc)
    
            c_out = ' '.join(commands)
            self.netlogo.command(c_out)
            
            logging.info(' -- Simulating period {0} of {1}'.format(period, self.run_length))
            #Run the NetLogo model for one tick
            self.netlogo.command('go')
            logging.debug('NetLogo step completed')

            #Create placeholder well list - required for MODFLOW WEL package if no wells active in NetLogo
            well_LRCQ_list = {}
            well_LRCQ_list[0] = [[0, 0, 0, 0]]
            ssm_data = {}
            ssm_data[0] = [[0, 0, 0, 0, itype['WEL']]]
            
            #Check the well agents which are active in NetLogo, and update the Python objects if required
            #The newgrid_flag indicates whether or not the grid should be recalculated to account for changes
            #in the list of active wells
            if well_obj_list:
                well_obj_list, newgrid_flag = update_runtime_objectlist(self.netlogo,
                                                                        well_obj_list,
                                                                        nl_read_well_attribs)
            
            if well_obj_list and newgrid_flag:
                #If the list of active wells has changed and if there are active wells, create a new grid object
                newgrid_obj = PyGrid()
                newgrid_obj.make_grid(well_obj_list, dmin=self.dmin, dmax=self.dmax,
                                      dz=self.dz, ztop=self.ztop, zbot=self.zbot,
                                      nstep=self.nstep, grid_extents=self.grid_extents)
                #Interpolate the temperature and head arrays to match the new grid 
                newgrid_obj.temp = grid_interpolate(grid_obj.temp[0,:,:], grid_obj, newgrid_obj)
                newgrid_obj.head = grid_interpolate(grid_obj.head[0,:,:], grid_obj, newgrid_obj)
                #Use the new simulation grid
                grid_obj = newgrid_obj
            
            logging.debug('Python update completed')

            if well_obj_list:
                for i in well_obj_list:
                    #Read well flows from NetLogo and locate each well in the simulation grid
                    i.Q = read_NetLogo_attrib(self.netlogo, 'Q', i.who)
                    i.calc_LRC(grid_obj)
                #Create well and temperature lists following MODFLOW/MT3DMS format
                well_LRCQ_list = create_LRCQ_list(well_obj_list, grid_obj)
                ssm_data = create_conc_list(well_obj_list)
                
            #Initialize MODFLOW packages using FloPy
            #ml = mf.Modflow(self.name, version='mf2005', exe_name=self.swtexe_name, model_ws=self.dirs[0])
            swtm = swt.Seawat(self.name, exe_name=self.swtexe_name, model_ws=self.dirs[0])  
            discret = mf.ModflowDis(swtm, nrow=grid_obj.nrow, ncol=grid_obj.ncol, nlay=grid_obj.nlay,
                                     delr=grid_obj.delr, delc=grid_obj.delc, laycbd=0, top=self.ztop, 
                                     botm=self.zbot, nper=self.nper, perlen=self.perlen, 
                                     nstp=self.nstp, steady=self.steady)
              
            bas = mf.ModflowBas(swtm, ibound=grid_obj.IBOUND, strt=grid_obj.head)
            lpf = mf.ModflowLpf(swtm, hk=self.HK, vka=self.VK, ss=0.0, sy=0.0, laytyp=0, layavg=0)
            
            wel = mf.ModflowWel(swtm, stress_period_data=well_LRCQ_list)
                
            words = ['head','drawdown','budget', 'phead', 'pbudget']
            save_head_every = 1
            oc = mf.ModflowOc(swtm)   
            pcg = mf.ModflowPcg(swtm, mxiter=200, iter1=200, npcond=1, 
                                hclose=0.001, rclose=0.001, relax=1.0, nbpol=0)
            #ml.write_input()
                
            #Initialize MT3DMS packages
            #mt = mt3.Mt3dms(self.name, 'nam_mt3dms', modflowmodel=ml, model_ws=self.dirs[0])
            adv = mt3.Mt3dAdv(swtm, mixelm=0,  #-1 is TVD
                              percel=1,
                              nadvfd=1,
                              #Particle based methods
                              nplane=0,
                              mxpart=250000,
                              itrack=3,
                              dceps=1e-4,
                              npl=5,
                              nph=8,
                              npmin=1,
                              npmax=16)
            btn = mt3.Mt3dBtn(swtm, cinact=-100., icbund=grid_obj.ICBUND, prsity=self.PEFF, sconc=[grid_obj.temp][0],
                              ifmtcn=-1, chkmas=False, nprobs=0, nprmas=1, dt0=0.0, ttsmult=1.5,
                              ttsmax=20000., ncomp=1, nprs=nprs, timprs=timprs, mxstrn=9999)
            dsp = mt3.Mt3dDsp(swtm, al=self.al, trpt=trpt, trpv=trpv, dmcoef=dmcoef)
            rct = mt3.Mt3dRct(swtm, isothm=0, ireact=0, igetsc=0, rhob=rho_b)
            gcg = mt3.Mt3dGcg(swtm, mxiter=50, iter1=50, isolve=1, cclose=1e-3, iprgcg=0)
            ssm = mt3.Mt3dSsm(swtm, stress_period_data=ssm_data)
            #mt.write_input()
                
            #Initialize SEAWAT packages
            # mswtf = swt.Seawat(self.name, 'nam_swt', modflowmodel=ml, mt3dmsmodel=mt,
            #                    model_ws=self.dirs[0])         
            swtm.write_input()
                
            #Run SEAWAT
            #m = mswtf.run_model(silent=True)
            m = swtm.run_model(silent=True)
            logging.debug('SEAWAT step completed')
            
            #Copy Modflow/MT3DMS output to new files
            shutil.copyfile(os.path.join(self.dirs[0], self.name+'.hds'),
                            os.path.join(self.dirs[0], self.name+str(period)+'.hds'))
            shutil.copyfile(os.path.join(self.dirs[0], 'MT3D001.UCN'),
                            os.path.join(self.dirs[0], self.name+str(period)+'.UCN'))
                
            #Create head file object and read head array for next simulation period
            h_obj = bf.HeadFile(os.path.join(self.dirs[0], self.name+str(period)+'.hds'))
            grid_obj.head = h_obj.get_data(totim=self.perlen)
 
            #Create concentration file object and read temperature array for next simulation period
            t_obj = bf.UcnFile(os.path.join(self.dirs[0], self.name+str(period)+'.UCN'))
            grid_obj.temp = t_obj.get_data(totim=self.perlen)
            
            logging.debug('Output processed')
            
            if well_obj_list:
                for i in well_obj_list:
                #Update each active Python well object with the temperature and head at its grid location
                    i.T_modflow = grid_obj.temp[i.L[0],i.R,i.C]
                    i.H_modflow = grid_obj.head[i.L[0],i.R,i.C]
                #Update the NetLogo agents from the corresponding Python objects
                write_NetLogo_attriblist(self.netlogo, well_obj_list, nl_update_well_attribs)
            
            #As an example of data exchange, we can calculate the fraction of the simulated grid in which
            #the temperature change is significant, and send this value to a NetLogo global variable
            use = subsurface_use(grid_obj, grid_obj.temp)

            write_NetLogo_global(self.netlogo, 'SubsurfaceUse', use)

            logging.debug('NetLogo update completed')
                
            h_obj.file.close()
            t_obj.file.close()
        
        self.netlogo.command('file-close-all')
        self._handle_outcomes()
            
            
    def _handle_outcomes(self):
        '''
        Method for reading NetLogo results into model output
        :param fns: dict with outcome name as key, and NetLogo output filename as value
        '''

        # for key, value in fns.items():
        #     with open(value) as fh:
        #         result = fh.readline()
        #         result = result.strip()
        #         result = result.split()
        #         result = [float(entry) for entry in result]
        #         self.output[key] = np.asarray(result)
        #     os.remove(value) 
           
        results = {}
        for key, value in self.fns.items():
            with open(value) as fh:
                result = fh.readline()
                result = result.strip()
                result = result.split()
                result = [float(entry) for entry in result]
                results[key] = np.asarray(result)
            os.remove(value)

        # temp_output = {}
        # for outcome in self.outcomes:
        #     varname = outcome.variable_name
        #     if len(varname)==1:
        #         varname = varname[0]
        #         temp_output[outcome.name] = results[varname]
        #     else:
        #         temp_output[outcome.name] = [results[var] for var in varname]

        # return temp_output
        
        return results




    def retrieve_output(self, method='mean'):
        '''
        Method for retrieving output after a model run.
        :param method: numpy function to process replication data. Default value stores the mean
                       of all replications - set to None to store all replications
        :returns: output of a model run - dict with outcome names as keys and time series as values
        '''
         
        return self.output

        
    
    def cleanup(self, seawat_cleanup=False):
        '''
        This model is called after finishing all the experiments, but 
        just prior to returning the results. This method gives a hook for
        doing any cleanup, such as closing applications. 
        
        In case of running in parallel, this method is called during 
        the cleanup of the pool, just prior to removing the temporary 
        directories.
        :param seawat_cleanup: boolean - delete SEAWAT output files after run 
        '''
        
        self.netlogo.kill_workspace()
        jpype.shutdownJVM()
        
        if seawat_cleanup:
            os.chdir(self.dirs[0])
            for name in glob.glob('*.*'):
                os.remove(name)
            os.chdir(self.wd)

class ToyModel(SingleReplication, ToyModel_MSI):
    pass
