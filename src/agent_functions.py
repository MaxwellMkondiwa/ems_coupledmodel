'''
Agent classes and functions for the Flopy/NetLogo coupled sandbox model.
These functions provide the main interface between NetLogo and Python
'''

import numpy as np
import pandas as pd
from scipy import interpolate
import numexpr as ne
import logging

import flopy.mt3d as mt3

#Variables (e.g. geographic coordinates) to be scaled between NetLogo and Python/SEAWAT.
#A scale factor of 10 indicates the NetLogo variables are scaled down by 10 relative
#to the Python/SEAWAT variables (allows for a smaller NL world to increase performance)
global scale_variables, scale_factor, scale_x, scale_y, offset_x, offset_y
scale_variables = ['xcor','ycor','plot_xmin','plot_xmax','plot_ymin','plot_ymax',
                   'building_xmin','building_xmax','building_ymin',
                   'building_ymax', 'Rth']

scale_factor = 10



class PyAgent(object):
    '''
    Super class for Python agent objects, do not instantiate directly
    '''
    
    def __init__(self):
        '''
        Initial properties common to all agents
        '''
        
        self.breed = ''
        self.who = 0
        self.xcor = 0
        self.ycor = 0
        self.localized = False
        
        
    def create_NetLogo_agent(self, netlogo, netlogo_update_attribs=None):
        '''
        Create an agent in NetLogo which corresponds to the Python agent object
        
        :param netlogo: NetLogoLink object
        :param netlogo_update_attribs: list of strings - attributes to be passed from the Python
                                       agent object to the NetLogo agent
        '''
        
        beginstr = ['create-', self.breed, 's 1']
        command = ''.join(beginstr)
        netlogo.command(command)
        
        if netlogo_update_attribs:
            self.update_NetLogo_agent(netlogo, netlogo_update_attribs)
            
    #@profile    
    def update_NetLogo_agent(self, netlogo, netlogo_update_attribs):
        '''
        Ask an existing agent in NetLogo to update properties from corresponding Python object.
        The write_NetLogo_attriblist function should be used for better performance when
        updating a set of agents

        :param netlogo: NetLogoLink object
        :param netlogo_update_attribs: list of strings - attributes to be passed from the Python
                                       agent object to the NetLogo agent
        '''
        
        beginstr = ['ask ', self.breed, ' {0} ['.format(self.who)]
        begin = ''.join(beginstr)
        midstr = []
        for attrib in netlogo_update_attribs:
            if attrib in scale_variables:
                value = getattr(self, attrib) / scale_factor
            else:
                value = getattr(self, attrib)
            if not isinstance(value, str):
                midstr.extend(('set ', attrib, ' {0} '.format(value)))
            else:
                midstr.extend(('set ', attrib, ' "'+value+'"'))
        endstr = ']'
        mid = ''.join(midstr)
        command = ''.join((begin, mid, endstr))
        netlogo.command(command)
        
        
    def update_Python_object(self, netlogo, netlogo_read_attribs):
        '''
        Ask an existing object in Python to read properties from the corresponding NetLogo agent
        
        :param netlogo: NetLogoLink object
        :param netlogo_read_attribs: list of strings - attributes to be passed from the NL
                                     agent to the corresponding Python object
        '''
        
        for attrib in netlogo_read_attribs:
            value = read_NetLogo_attrib(netlogo, attrib, self.who, self.breed)
            if attrib in scale_variables:
                value = value * scale_factor

            setattr(self, attrib, value)
        

class PyWell(PyAgent):
    '''
    Python well object class
    '''
        
    def __init__(self, external_attributes=None):
        '''
        Initialize a Python well object
        
        :param external_attributes: dict of attributes to be assigned to the object (generated
                                    from Excel or NetLogo)
        '''
        
        super(PyWell, self).__init__()
        
        self.breed = 'well'
        self.L = 0
        self.R = 0
        self.C = 0
        self.Q = 0
        self.localized = True
        self.Tmodflow = 0
        self.Hmodflow = 0
        self.Tqmin_c = 0
        self.Tqmin_h = 0
        self.setpointyear = 0
        
        if external_attributes:
            for attrib, value in external_attributes.items():
                setattr(self, attrib, value)    
        
        
    #@profile
    def calc_setpoints(self, T_series, year, nYr=5):
        '''
        Calculate cooling and heating setpoints which balance storage over a given period
        for a given well object
        
        Based on adaptSetPoints.m by Theo Olsthoorn as implemented in mfLab
        
        :param T_series: pandas series of daily temperatures, indexed by day
        :param year: int - year for which to calculate the setpoints
        :param nYr: int - number of years over which the storage should be balanced
        '''
        
        Tqmax_c = self.Tqmax_c #Temporary variables to enable the use of numexpr functions
        Tqmax_h = self.Tqmax_h
        
        #Partial temperature series over which the setpoints should be computed
        calibration_temp = T_series[str(year - nYr):str(year - 1)]
        
        T0_c = 0
        T0_h = 20
        
        tqmc1 = self.Tqmax_c
        tqmc2 = T0_c
        tqmc = 0.5 * (tqmc1 + tqmc2)
         
        while tqmc1 - tqmc2 > 0.01:
     
            QdCooling = self.QdMax * ne.evaluate('calibration_temp > tqmc') * \
                np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                           ne.evaluate('(calibration_temp-tqmc) / (Tqmax_c-tqmc)'))
            QyCooling = np.sum(QdCooling) / nYr
     
            if QyCooling < self.Qy:
                tqmc1 = tqmc
                tqmc = 0.5 * (tqmc1 + tqmc2)
            else:
                tqmc2 = tqmc
                tqmc = 0.5 * (tqmc1 + tqmc2)
     
        self.Tqmin_c = tqmc
        
        tqmh1 = self.Tqmax_h
        tqmh2 = T0_h
        tqmh = 0.5 * (tqmh1 + tqmh2)
     
        while abs(tqmh1 - tqmh2) > 0.01:
            QdHeating = self.QdMax * ne.evaluate('calibration_temp < tqmh') * \
                np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                           ne.evaluate('(calibration_temp-tqmh) / (Tqmax_h-tqmh)'))
            QyHeating = np.sum(QdHeating) / nYr
            if QyHeating < self.Qy:
                tqmh1 = tqmh
                tqmh = 0.5 * (tqmh1 + tqmh2)
            else:
                tqmh2 = tqmh
                tqmh = 0.5 * (tqmh1 + tqmh2)
 
        self.Tqmin_h = tqmh
        self.setpointyear = year
                
                 
    #@profile     
    def calc_flow(self, T_period, imb):
        '''
        Calculate the average daily flow of a well object, as required by
        the Modflow WEL package. Use once the setpoints have been computed
        
        :param T_period: pandas series of daily temperatures over which the flows
                         should be computed
        '''
        
        Tqmin_c = self.Tqmin_c #Temporary variables for numexpr functions
        Tqmax_c = self.Tqmax_c
        Tqmin_h = self.Tqmin_h
        Tqmax_h = self.Tqmax_h
        
        QdCooling = self.QdMax * ne.evaluate('T_period > Tqmin_c') * \
            np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                       ne.evaluate('(T_period-Tqmin_c) / (Tqmax_c-Tqmin_c)'))
 
        QdHeating = self.QdMax * ne.evaluate('T_period < Tqmin_h') * \
            np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                       ne.evaluate('(T_period-Tqmin_h) / (Tqmax_h-Tqmin_h)'))
             
        if self.IsCold == 1:
            Qday = QdHeating - QdCooling
        else:
            Qday = QdCooling - QdHeating
     
        self.Q = np.mean(Qday)

        if (self.Q < 0 and self.IsCold == 1): #Summer cold
            self.Q = self.Q * (1 + imb)
        elif (self.Q > 0 and self.IsCold == 0): #Summer warm
            self.Q = self.Q * (1 + imb)
        elif (self.Q < 0 and self.IsCold == 0): #Winter warm
            self.Q = self.Q * (1 - imb)
        elif (self.Q > 0 and self.IsCold == 1): #Winter cold
            self.Q = self.Q * (1 - imb)
        
        
    #@profile   
    def calc_LRC(self, grid_obj):
        '''
        Calculate the layer, row and column position of a well object within a grid object
        
        :param grid_obj: Python grid object in which the well is to be located
        '''
        
        #self.L = 0 #For sandbox model
        
        self.R = int(len(grid_obj.YGR) - np.nonzero(grid_obj.YGR[::-1] > self.ycor)[0][0]) - 1
        
        self.C = int(np.nonzero(grid_obj.XGR > self.xcor)[0][0]) - 1
        zgr_lay = grid_obj.botm[:, self.R, self.C]
        lay_id = np.arange(len(zgr_lay))
        self.zgr = np.insert(zgr_lay, 0, grid_obj.top[self.R, self.C]) #Vector of vert. grid coords at location of well

        zrange = np.array([self.z0, self.z0 - self.FilterLength])

        try:
            self.stop_idx = np.where((self.zgr <= zrange[1]))[0][0]
        except:
            self.stop_idx = lay_id[-1]
        self.start_idx = np.where((self.zgr >= zrange[0]))[0][-1]
        #lay_idx = np.s_[start_idx:stop_idx]
        self.L = lay_id[self.start_idx:self.stop_idx] #Vector of layer IDs
        if len(self.L) == 0:
            self.L = lay_id[0:]
        
        
class PySystem(PyAgent):
    '''
    Python ATES system object class
    '''
        
    def __init__(self, external_attributes=None):
        '''
        Initialize a Python ATES system object
        
        :param external_attributes: dict of attributes to be assigned to the object (generated
                                    from Excel or NetLogo)
        '''
        
        super(PySystem, self).__init__()
        
        if external_attributes:
            for attrib, value in external_attributes.items():
                setattr(self, attrib, value)     

        self.breed = 'system'
        




class PyGrid(object):
    '''
    Instantiate a Python grid object
    '''
     
    def __init__(self):
     
        self.XGR = []
        self.YGR = []
        self.ncol = 0
        self.nrow = 0
        self.nlay = 0
        self.delr = []
        self.delc = []
        self.IBOUND = []
        self.ICBUND = []
        self.head = np.array([np.ones((1,1))])
        self.temp = np.array([np.ones((1,1))])
        self.salinity = np.array([np.ones((1,1))])
 
 
    def make_grid(self, well_obj_list, ztop, zbot, aroundAll=500, dmin=5, dmax=20, dz=5, nstep=2, grid_extents=None):
        '''
        Update the properties of a grid object - based on makegrid.m by Ruben Calje, 
        Theo Olsthoorn and Mark van der Valk as implemented in mfLab
         
        :param well_obj_list: list of Python well objects
        :param aroundAll: extra grid allowance around the "bounding box" of well coordinates
        :param dmin: target for minimum grid cell size
        :param dmax: target for maximum grid cell size
        :param nstep: refinement factor for grid cells around wells
        :param grid_extents: list of coordinates in the format [min_x, max_x, min_y, max_y]
                             If omitted, grid extents are calculated dynamically based on well coordinates
                             and aroundAll parameter
        '''
         
        wells_xy = np.array([[i.xcor, i.ycor] for i in well_obj_list]) 
         
        xw = np.ceil(wells_xy[:,0] / dmin) * dmin
        yw = np.ceil(wells_xy[:,1] / dmin) * dmin
         
        if grid_extents: 
            min_x = grid_extents[0]
            max_x = grid_extents[1]
            min_y = grid_extents[2]
            max_y = grid_extents[3]
        else:
            min_x = np.min(xw - aroundAll)
            max_x = np.max(xw + aroundAll)
            min_y = np.min(yw - aroundAll)
            max_y = np.max(yw + aroundAll)      
         
        XGR = np.arange(min_x, max_x + dmin, dmax)
        YGR = np.arange(min_y, max_y + dmin, dmax)
         
        dx = np.logspace(np.log10(dmin), np.log10(dmax),nstep)
        d = np.cumsum(np.append(dx[0] / 2, dx[1:len(dx)]))
        L = d[-1]
        subgrid = np.append(-d[::-1], d)
         
        for iW in range(len(wells_xy)):
            XGR = XGR[(XGR < wells_xy[iW,0] - L) | (XGR > wells_xy[iW,0] + L)]
            YGR = YGR[(YGR < wells_xy[iW,1] - L) | (YGR > wells_xy[iW,1] + L)]
             
        Nx = len(XGR);
        Ny = len(YGR);
        Ns = len(subgrid);
        Nw = len(wells_xy);
         
        XGR = np.append(XGR, np.zeros(Nw*Ns))
        YGR = np.append(YGR, np.zeros(Nw*Ns))
         
        for iW in range(len(wells_xy)):
            XGR[Nx + iW*Ns + np.arange(0,Ns)] = wells_xy[iW,0] + subgrid;
            YGR[Ny + iW*Ns + np.arange(0,Ns)] = wells_xy[iW,1] + subgrid;
         
        #XGR, YGR: 1D arrays of cell coordinates (respectively columns and rows)
        self.XGR = cleangrid(np.unique(np.around(XGR*100)/100), dmin)
        self.YGR = cleangrid(np.unique(np.around(YGR*100)/100), dmin)[::-1]

        self.ncol = len(self.XGR) - 1 #Number of grid columns
        self.delr = np.diff(self.XGR) #Width of each column
        self.nrow = len(self.YGR) - 1 #Number of grid rows
        self.delc = -np.diff(self.YGR) #Height of each row

        self.top = ztop * np.ones([self.nrow, self.ncol])
        botm_range = np.arange(zbot, ztop, dz)[::-1]
        botm_2d = np.ones([self.nrow, self.ncol])
        self.botm = botm_2d*botm_range[:, None, None]
        self.nlay = len(botm_range)

        self.IBOUND, self.ICBUND = boundaries(self) #Create grid boundaries
             


def update_runtime_objectlist(netlogo, obj_list, nl_read_attribs, breed=None, objclass=None):
    '''
    Compare the active NetLogo agents with the existing Python agent object list, and create/remove
    corresponding Python objects as needed
    
    :param netlogo: NetLogoLink object
    :param obj_list: list of Python objects (of uniform type) to compare with the NetLogo agents
    :param nl_read_attribs: list of strings - attributes to be read from NetLogo when creating new Python objects
    :returns: updated list of Python objects
    :returns: boolean flag indicating whether the grid needs to be recalculated to account for new local agents
    '''    
    
    newgrid_flag = False #By default, do not recalculate the grid
     
    if obj_list:
        breed = obj_list[0].breed
        objclass = obj_list[0].__class__
        if hasattr(obj_list[0], 'L') or hasattr(obj_list[0], 'R'): 
            #Assumes localized agents have an attribute for layer and/or row-column location
            localized_agent = True
    else:
        # breed = breed
        # objclass = objclass
        localized_agent = True
    
    #Check the agents which are active in NetLogo        
    netlogo_active_agents = read_NetLogo_attriblist(netlogo, 
                                                    attribute='who', 
                                                    breed=breed)
    logging.debug('NetLogo active wells:')
    logging.debug(netlogo_active_agents)

    #Check the well objects which are active in Python
    python_active_agents = [i.who for i in obj_list]
    
    #Check any well objects to be created in Python to match the NetLogo agents
    who_new_agents = list(set(netlogo_active_agents) - set(python_active_agents))
    logging.debug('Python wells to be created:')
    logging.debug(who_new_agents)
    
    #Check any well objects to be removed in Python to match the NetLogo agents
    who_removed_agents = list(set(python_active_agents) - set(netlogo_active_agents))
    logging.debug('Python wells to be removed:')
    logging.debug(who_removed_agents)
        
    if who_new_agents:
        #Create new Python well objects and update the list of active well objects
        new_obj_list = createobjfromNetLogo(netlogo,
                                            objclass,
                                            nl_read_attribs,
                                            who_new_agents)
        obj_list += new_obj_list
        if localized_agent:
            newgrid_flag = True #Localized agent list has changed - update the grid
 
    if who_removed_agents:
        #Remove Python well objects from the list of active well objects
        obj_list = [i for i in obj_list if i.who not in who_removed_agents]
        if localized_agent:
            newgrid_flag = True
        
    logging.debug('Python active wells:')
    logging.debug([i.who for i in obj_list])

    return obj_list, newgrid_flag


def createobjfromExcel(objclass, filename, sheetname, cols):
    '''
    Create a list of Python agent objects based on data from an Excel configuration file
    
    :param objclass: Python object class to be instantiated
    :param filename: .xls to be opened
    :param sheetname: sheet containing the object data
    :param cols: column indices containing the data to be read - ensure the columns
                 match the netlogo_init_attribs list to maintain consistency between
                 Excel, Python and NetLogo
    :returns: list of Python agent objects, instantiated using the Excel attributes
    '''
    
    excel_df = pd.read_excel(filename, sheetname, parse_cols=cols)
    obj_list = []
     
    for i in range(len(excel_df)):
        obj_attributes = dict(excel_df.iloc[i])
        newobj = objclass(obj_attributes)
        obj_list.append(newobj)   

    return obj_list

#@profile
def createobjfromNetLogo(netlogo, objclass, netlogo_read_attribs, wholist):
    '''
    Create a list of Python agent objects corresponding to NetLogo agents
    
    :param netlogo: NetLogoLink object
    :param objclass: Python object class to be instantiated
    :param netlogo_read_attribs: list of strings - attributes to be read from NetLogo
    :param wholist: list of NetLogo 'who's for the wells to be created
    :param scale_factor: scale factor between the NetLogo world and real coordinates
    :returns: list of Python well objects, instantiated using the NetLogo attributes
    '''
    
    obj_list = []

    for i in wholist:
        values = []
        for attrib in netlogo_read_attribs:
            value = read_NetLogo_attrib(netlogo, attrib, i)
            if attrib in scale_variables:
                value = value * scale_factor

            values.append(value)
            
        obj_attributes = dict(zip(netlogo_read_attribs, values))
        newobj = objclass(obj_attributes)
        obj_list.append(newobj)  
        
    return obj_list


def read_NetLogo_variable(netlogo, variable):
    '''
    Report a variable from NetLogo
    
    :param netlogo: NetlogoLink object
    :param variable: string - variable to report from NetLogo
    :returns: float - value of the attribute as reported by NetLogo
    '''
    
    value = netlogo.report(variable)
    
    return value 


def read_NetLogo_attrib(netlogo, attribute, who, breed='turtle'):
    '''
    Read a given attribute from a single NetLogo agent
    
    :param netlogo: NetlogoLink object
    :param attribute: string - attribute to read from NetLogo
    :param who: NetLogo who of the agent
    :param breed: string - NetLogo breed of the agent
    :returns: float - value of the attribute as reported by NetLogo
    '''
    
    beginstr = ['[', attribute, '] of ', breed, ' {0}'.format(who)]
    beginstr = ''.join(beginstr)
    value = netlogo.report(beginstr)
    
    return value    
    

def read_NetLogo_attriblist(netlogo, attribute, breed='turtle', with_property='who >= 0'):
    '''
    Read a given attribute from a set of NetLogo agents
    
    :param netlogo: NetLogoLink object
    :param attribute: string- attribute to read from NetLogo
    :param breed: string - NetLogo breed of the agents
    :param with_property: string - if omitted, reads from all agents of a given breed
    :returns: list of floats corresponding to the agent attributes reported by NetLogo
    '''
    
    beginstr = ['[', attribute, '] of ', breed, 's with [', with_property,']']
    beginstr = ''.join(beginstr)
    try:
        values = netlogo.report(beginstr)
        attribute_list = list(values)
    except:
        attribute_list = []
    
    return attribute_list



#@profile
def write_NetLogo_attriblist(netlogo, obj_list, netlogo_update_attribs):
    '''
    Update a set of NetLogo agents with a list of attributes from a list of Python objects
    
    :param netlogo: NetLogoLink object
    :param obj_list: list of Python objects (of uniform type) for which to update corresponding NL agents
    :param netlogo_update_attribs: list of strings - attributes to be passed from the Python
                                   agent objects to the NetLogo agents
    '''

    breed = obj_list[0].breed
    wholist =  [i.who for i in obj_list]
    attriblist = []
    for attrib in netlogo_update_attribs:
        values = [getattr(i, attrib) for i in obj_list]
        if attrib in scale_variables:
            values = [val / scale_factor for val in values]
        elif attrib == 'xcor':
            values = [(val - offset_x) / scale_x for val in values]
        elif attrib == 'ycor':
            values = [(val - offset_y) / scale_y for val in values]
        attriblist.append(values)
    
    whostr = ' '.join(map(str, wholist))
    attribstr = []
    for i in range(len(attriblist)):
        values = ' '.join(map(str, attriblist[i]))
        liststr=['[', values, ']']
        attribstr.append(''.join(liststr))
    attribstr = ' '.join(attribstr)
       
    askstr = []
    setstr = []
    for i in range(len(attriblist)):
        askstr.extend(('?{0} '.format(i+2)))
        setstr.extend(('set ', netlogo_update_attribs[i], ' ?{0} '.format(i+2)))
    askstr = ''.join(askstr)
    setstr = ''.join(setstr)
    
    commandstr = ['(foreach [', whostr, '] ', attribstr, ' [ [?1 ', askstr, '] -> ask ', breed, ' ?1 [', setstr,']])']
    commandstr = ''.join(commandstr)

    netlogo.command(commandstr)


def write_NetLogo_global(netlogo, variable, value):
    '''
    Update a global variable in NetLogo

    :param netlogo: NetLogoLink object
    :param variable: string - NetLogo global variable to be updated
    :param value: value to which the global variable will be set
    '''
    
    globalstr = []
    if variable in scale_variables:
        value = value / scale_factor
    globalstr.extend(('set ', variable, ' {0} '.format(value)))
    commandstr = ''.join(globalstr)
    
    netlogo.command(commandstr)


def create_conc_list(well_obj_list, attrib='T_inj'):
    '''
    Output a species concentration array as required for MT3DMS. Default attribute is temperature
     
    :param well_obj_list: list of Python well objects
    :param attrib: Python object attribute corresponding to the requested concentration
    :returns: array of concentrations, formatted for the MT3DMS SSM package
    '''
    itype = mt3.Mt3dSsm.itype_dict()
    ssm_data = {}
    ssmlist = []
    if isinstance(attrib,list):
        for i in well_obj_list:
            n_layers = len(i.L)
            if n_layers == 1:
                ssmlist.append([i.L, i.R, i.C, getattr(i, attrib[0]), itype['WEL'], \
                                getattr(i, attrib[0]), getattr(i, attrib[1])])
            else:
                for k in range(n_layers):
                    ssmlist.append([i.L[k], i.R, i.C, getattr(i, attrib[0]), itype['WEL'], \
                                    getattr(i, attrib[0]), getattr(i, attrib[1])])                    
                
    elif isinstance(attrib,str):
        for i in well_obj_list:
            n_layers = len(i.L)
            if n_layers == 1:
                ssmlist.append([i.L, i.R, i.C, getattr(i, attrib), itype['WEL']])
            else:
                for k in range(n_layers):
                    ssmlist.append([i.L[k], i.R, i.C, getattr(i, attrib), itype['WEL']])
    
    ssm_data[0] = ssmlist

    return ssm_data
 
 
def create_LRCQ_list(well_obj_list, grid_obj):
    '''
    Format a list of layer/row/column positions and flows
     
    :param well_obj_list: list of Python well objects
    :returns: LRCQ list, formatted for the Modflow WEL package
    '''
     
    LRCQ_dict = {}
    LRCQ_list = []
    for i in well_obj_list:
        n_layers = len(i.L)
        if n_layers == 1:
            LRCQ_list.append([i.L[0], i.R, i.C, i.Q])
        else:
            trans_vec = -np.diff(i.zgr[i.start_idx:i.stop_idx+1])*grid_obj.HK[:,i.R,i.C][i.start_idx:i.stop_idx]
            q_vec = i.Q*trans_vec/np.sum(trans_vec)
            for k in range(n_layers):
                LRCQ_list.append([i.L[k], i.R, i.C, q_vec[k]])
    LRCQ_dict[0] = LRCQ_list
    
    return LRCQ_dict


def subsurface_use(grid_obj, t_array, Tcrit=1, Tref=10, bounds=None):
    '''
    Calculate the fraction of subsurface area which is "in use" for energy storage -
    based on Analyze_Extr.m by Martin Bloemendal
     
    :param grid_obj: Python grid object
    :param t_array: 2D temperature array
    :param Tcrit: float - deltaT above which the temperature change is considered significant
    :param Tref: float - reference initial temperature of the subsurface
    :param bounds: list of grid extents to use for calculation (MinL, MaxL, MinR, MaxR, MinC, MaxC)
    :returns: float - fraction of the area of interest in which the deltaT is significant
    '''
           
    if bounds:
        MinL, MaxL, MinR, MaxR, MinC, MaxC = bounds
    else:
        MinL = 0
        MaxL = grid_obj.nlay
        MinR = 0
        MaxR = grid_obj.nrow
        MinC = 0
        MaxC = grid_obj.ncol
     
    total_vol = cell_volumes(grid_obj) 
    reference_vol = total_vol[MinL:MaxL,MinR:MaxR, MinC:MaxC]
    reference_temp = t_array[MinL:MaxL, MinR:MaxR, MinC:MaxC] - Tref
     
    area_w = (reference_temp > Tcrit) * reference_vol
    area_c = (reference_temp < -Tcrit) * reference_vol
    used_fraction = (np.sum(area_w)+np.sum(area_c)) / np.sum(reference_vol)
     
    return used_fraction


def cleangrid(XGR, dmin):
    '''
    Remove cells smaller than dmin in a grid object. Based on Ruben Calje, 
    Theo Olsthoorn and Mark van der Valk
     
    :param XGR: 1D array of grid coordinates
    :param dmin: float - target for minimum grid cell size
    :returns: updated 1D array of grid coordinates
    '''
     
    k=0
    while 1:
        Dx = np.diff(XGR);
        minDx = np.minimum(Dx[:len(Dx)-1], Dx[1:])
        minminDx = np.amin(minDx)
 
        if np.fmod(k, 2) == 0:
            imin = np.nonzero(minDx == minminDx)[0][0]
        else:
            imin = np.nonzero(minDx == minminDx)[0][-1]
 
        if minminDx < dmin:
            XGR = np.delete(XGR, imin+1)
            k += 1
        else:
            return XGR
 
 
def boundaries(grid_obj):
    '''
    Create boundary lists for a grid object. Configured to yield a boundary for heads and
    concentrations on the edges of the grid
     
    :param nrow: int - number of grid rows
    :param ncol: int - number of grid columns
    :returns: nested lists representing the 2D boundary arrays (as required for Modflow/MT3DMS)
    '''
     
    ib = -np.ones((grid_obj.nlay, grid_obj.nrow, grid_obj.ncol))
    ib[:,1:-1,1:-1] = 1

    IBOUND = ib
    ICBUND = IBOUND


    return IBOUND, ICBUND 
 


def cell_volumes(grid_obj):
    '''
    Calculate the area of cells in the grid
     
    :param grid_obj: Python grid object
    :returns: 2D array of cell areas
    '''
    

    Dx = np.diff(grid_obj.XGR)
    Dy = -np.diff(grid_obj.YGR)
     
    area_array = np.zeros((len(Dy), len(Dx)))
    for i in range(len(Dy)):
        for j in range(len(Dx)):
            area_array[i,j] = Dy[i]*Dx[j]
    
     
    #vol_array = np.zeros((grid_obj.nlay, grid_obj.nrow, grid_obj.ncol))

    zgr = np.insert(grid_obj.botm, 0, grid_obj.top, axis=0)
    dz = np.diff(zgr, axis=0)
    vol_array = dz*area_array

    return vol_array
     
 
def grid_interpolate(olddata_array, oldgrid_obj, newgrid_obj):
    '''
    Interpolates a 2D data array to match the coordinates of a new grid object
     
    :param olddata_array: 2D array to be interpolated
    :param oldgrid_obj: reference Python grid object (corresponding to the input array)
    :param newgrid_obj: new Python grid object (over which the array should be interpolated)
    :returns: interpolated 2D array
    '''
     
    orig_x = oldgrid_obj.XGR[1:]
    orig_y = oldgrid_obj.YGR[1:][::-1] #Reverse Y axis to match Modflow row scheme
     
    new_x = newgrid_obj.XGR[1:]
    new_y = newgrid_obj.YGR[1:][::-1]
     
    interp_obj = interpolate.interp2d(orig_x, orig_y, olddata_array[::-1], kind='linear')
    newdata_grid = interp_obj(new_x, new_y)[::-1]
     
    return newdata_grid
