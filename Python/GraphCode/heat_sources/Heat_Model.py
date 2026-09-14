import numpy as np
import csv 
import datetime

import material_values as mat

class Path():
    def __init__(s, points):
        s.pts = points['pts'] # List of Tuples [(x1, y1), (x2, y2), ...]
        s.type = points['type'] # Reverse or Cycle 
        s.current_pt=s.pts[0]
        s.pos=0
        s.rev_toggle=False
    
    def CurrPt(s):
        return [s.current_pt[0], s.current_pt[1]]
    
    def NextPt(s, new=False):
        if s.rev_toggle: 
            s.pos=s.pos-1
        else:
            s.pos=s.pos+1

        if s.pos==len(s.pts) or s.pos==-1:

            if s.type=='Reverse':
                if s.rev_toggle:
                    s.rev_toggle=False
                    s.pos=s.pos+2
                else:
                    s.rev_toggle=True
                    s.pos=s.pos-2

            elif s.type=='Cycle':
                s.pos=0

            else: 
                ValueError('Wrong "type" input for Points.')

        s.current_pt=s.pts[s.pos]
        
        if new: 
            return s.CurrPt()

class Model(): 
    def __init__(s, material, substrate, pts, dwell, model, layers, filename):
        s._h_conv = 5

        s._material = material
        s._mat_props = mat.params[material] # a lot of stuff
        for i in ['Wire','Substrate']: # Calculate alpha and tau
            s._mat_props[i]['Thermal Diffusivity']=s._mat_props[i]['Thermal Conductivity']/(s._mat_props[i]['Density']*s._mat_props[i]['Specific Heat'])
            s._mat_props[i]['Tau'] = (s._mat_props[i]['Specific Heat']*(s._mat_props[i]['Density']*1000000000)) / s._h_conv
        s._mat_props['Welding Settings']['Traverse Speed'] = s._mat_props['Welding Settings']['Traverse Speed'] /60 * 25.4 # in/min -> mm/s

        s._substrate = substrate # length, width, height
        s._points=Path(pts)
        s._model=model # model type
        s._dwell_time = dwell # seconds
        s._layers = layers

        s._csv_file = open(f"{filename}.csv", mode="w", newline="")
        s._writer = csv.writer(s._csv_file)
        s._writer.writerow(["Time (s)", "Avg Temp (K)", "Min Temp (K)", "Max Temp (K)"])

        s._timer_tog = True

        print('---')
        print(f'Starting simulation: {filename}')

    def _get_emissivity(s): 
        if s._material == 'Aluminum':
            emissivity = np.zeros_like(s._temperature_grid, dtype=float)
            emissivity[s._temperature_grid<=60+273.15] = 0.3
            emissivity[s._temperature_grid<=80+273.15] = 0.255
            emissivity[s._temperature_grid<=100+273.15] = 0.25
            emissivity[s._temperature_grid<=140+273.15] = 0.205
            emissivity[s._temperature_grid<=180+273.15] = 0.2
            emissivity[s._temperature_grid>180+273.15] = 0.155
            emissivity[s._bead_mask==0] = 0 # Remove if not deposited
        elif s._material == 'Steel':
            emissivity = np.zeros_like(s._temperature_grid, dtype=float)
            emissivity[s._temperature_grid<=60+273.15] = 0.275
            emissivity[s._bead_mask==0] = 0 # Remove if not deposited

        return emissivity

    def _get_q(s, layer): 
        if layer==1: 
            try: 
                q = s._mat_props['Welding Settings']['Thermal Efficiency'] * s._mat_props['Welding Settings']['L1 Current'] * s._mat_props['Welding Settings']['L1 Voltage']
            except KeyError: 
                q = s._mat_props['Welding Settings']['Thermal Efficiency'] * s._mat_props['Welding Settings']['Current'] * s._mat_props['Welding Settings']['Voltage']
        else: 
            q = s._mat_props['Welding Settings']['Thermal Efficiency'] * s._mat_props['Welding Settings']['Current'] * s._mat_props['Welding Settings']['Voltage']
        return q

    def _get_wire_width(s, layer): 
        if layer==1: 
            try: 
                w = s._mat_props['Wire']['L1 Width']
            except KeyError: 
                w = s._mat_props['Wire']['Width']
        else: 
            w = s._mat_props['Wire']['Width']
        return w
    
    def _get_goldak_params(s, layer): 
        if layer==1: 
            try: 
                g = s._mat_props['L1 Goldak Params']
            except KeyError: 
                g = s._mat_props['Goldak Params']
        else: 
            g = s._mat_props['Goldak Params']
        return g
    
    def _get_substeps(s, pt1, pt2):        
        x1 = pt1[0]; y1=pt1[1] # mm
        x2 = pt2[0]; y2=pt2[1] # mm

        dist = int(np.sqrt(((x2-x1)**2)+((y2-y1)**2))) / s._dx

        dist = int(round(dist,0))

        step_pts = []
        for i in range(dist+1): 
            t = i / dist
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            step_pts.append([x, y])

        step_length = np.sqrt(((step_pts[1][0]-step_pts[0][0])**2)+((step_pts[1][1]-step_pts[0][1])**2))

        return step_pts, step_length

    def run(s):
        # =================
        # ==== INIT P2 ====
        # =================
        s._dx = 0.7 #mm

        s._dt = s._dx/(1/s._mat_props['Welding Settings']['Traverse Speed'])

        s._T_env = 25+273.15 # Kelvin

        xlength = int(round(s._substrate['length'],1)/s._dx)
        ylength = int(round(s._substrate['width'],1)/s._dx)
        zlength = int(round(s._substrate['thickness']+(s._layers * s._mat_props['Wire']['Height']),1)/s._dx)

        voxel_sizes = np.array([
            s._substrate['length']/xlength,
            s._substrate['width']/ylength,
            (s._substrate['thickness']+(s._layers * s._mat_props['Wire']['Height']))/zlength,
            ])
        
        print(f'=> Voxel Sizes are ({str(round(voxel_sizes[0],3))}, {str(round(voxel_sizes[1],3))}, {str(round(voxel_sizes[2],3))}) with s._dx = {s._dx} mm.')
        
        s._geometric_size = voxel_sizes * np.array([xlength, ylength, zlength])

        s._temperature_grid = np.full(
            (xlength, ylength, zlength),
            s._T_env,
            dtype=float
        )

        substrate_idx = int(round(s._substrate['thickness'],1)/s._dx)

        s._bead_mask = np.full(
            (xlength, ylength, zlength),
            0,
            dtype=float
        )

        s._bead_mask[:,:,:substrate_idx]=2

        """
        Use np.where to do operations where the s._bead_mask has a specific value.
        
        >>> a = np.array([[0, 1, 2],
                          [0, 2, 4],
                          [0, 3, 6]])
        >>> np.where(a < 4, a, -1)  # -1 is broadcast
        array([[ 0,  1,  2],
               [ 0,  2, -1],
               [ 0,  3, -1]])
        """

        # =================
        # === RUN MODEL ===
        # =================
        
        log_time = 0
        for layer in range(1,s._layers+1):
            print(f'==> Starting layer {layer} using {s._model} Method.')

            # Find z location of torch (doesn't change with x, y)
            z = s._substrate['thickness']+(layer * s._mat_props['Wire']['Height'])

            H_i = s._get_q(layer)

            substeps, steplength = s._get_substeps(s._points.CurrPt(), s._points.NextPt(True))

            for x, y in substeps: 
                if s._timer_tog: 
                    timer_start = datetime.datetime.now() 

                if s._model=='Rosenthal': 
                    # Heating 
                    s._rosenthal_applyheat(H_i, steplength, layer, x, y, z)

                    # Diffusing updates s._temperature_grid
                    s._rosenthal_diffuse()

                    # Cooling 
                    s._rosenthal_cooling()                    
                elif s._model=='Goldak': 
                    # This happens for every step so just put it here
                    rho_field = np.zeros_like(s._bead_mask, dtype=float)
                    cp_field = np.zeros_like(s._bead_mask, dtype=float)
                    for num,i in enumerate(['Wire','Substrate']):
                        rho_field[s._bead_mask == num+1] = s._mat_props[i]['Density']
                        cp_field[s._bead_mask == num+1] = s._mat_props[i]['Specific Heat']
                    rho_field[s._bead_mask==0]= 1.17/1000000000 #air in Atlanta.... kg/m^3 -> kg/mm^3
                    cp_field[s._bead_mask==0]=1005 #air, J/kg-K
                    s._rho_cp_field = rho_field * cp_field

                    # Heating 
                    s._goldak_applyheat(H_i, layer, x, y, z)

                    # Diffusing updates s._temperature_grid
                    s._goldak_diffuse()

                    # Cooling 
                    s._goldak_cooling(layer, x, y)

                if s._timer_tog: 
                    timer_end = datetime.datetime.now() 
                    timer_diff = (timer_end-timer_start).total_seconds()
                    time_rem = s._layers*len(substeps)*timer_diff #seconds
                    time_rem_h = int(np.floor(time_rem/60/60))
                    time_rem_m = int(np.floor(time_rem/60) - time_rem_h*60)
                    time_rem_s = round(time_rem - time_rem_m*60 - time_rem_h*60*60,3)
                    print(f'----*Estimated {time_rem_h}:{time_rem_m}:{time_rem_s} (hh:mm:ss) time remaining for all layers.')
                    s._timer_tog = False

                # =================
                # ==== LOGGING ====
                # =================

                s._writer.writerow([round(log_time,2),
                    float(np.mean(s._temperature_grid)),
                    float(np.min(s._temperature_grid)),
                    float(np.max(s._temperature_grid))])
                
                log_time += s._dt

        print('Done!\n=====\n=====')
        s._csv_file.close()

    # ========================================================
    # ================ GOLDAK MODEL EQUATIONS ================
    # ========================================================
    
    def _goldak_applyheat(s, H_i, layer, Tx, Ty, Tz):
        # Assume x = Torch direction, y = Torch orthogonal, z = Torch height

        params = s._get_goldak_params(layer)
        # 'Goldak Params': ['b', 'c_1', 'c_2', 'f_f', 'f_r']
        
        a = params['a']                      # cm, Width of heat source
        b = params['b']                      # cm, Depth of penetration
        c1=params['c_1']; c2=params['c_2']   # cm, Front length and tail length, respectively
        f_f=params['f_f']; f_r=params['f_r'] # Note f_f + f_r = 2

        v = (s._mat_props['Welding Settings']['Traverse Speed']/1000) # mm/s -> m/s

        tau = c1 / v # lag factor
        # https://www.cfd-online.com/Forums/ansys/205944-two-simultaneous-heat-sources-ansys-apdl.html
        # Literally no one else cites what this value is related to

        # Pull subset of idx to localize
        pad = 5 # mm
        xmin = (Tx - (c2+pad)); xmax = (Tx + (c1+pad))  
        ymax, ymin = Ty + np.array([1, -1]) * (a+pad)
        zmax, zmin = Tz + np.array([1, -1]) * (b+pad)

        # Constrain to bounds of model
        if xmin<0: 
            xmin=0
        if ymin<0: 
            ymin=0
        if zmin<0: 
            zmin=0
        if xmax>(np.array(list(np.shape(s._temperature_grid)))*np.array([s._dx, s._dx, s._dx]))[0]: 
            xmax = np.array(list(np.shape(s._temperature_grid)))*np.array([s._dx, s._dx, s._dx])[0]
        if ymax>(np.array(list(np.shape(s._temperature_grid)))*np.array([s._dx, s._dx, s._dx]))[1]: 
            ymax = np.array(list(np.shape(s._temperature_grid)))*np.array([s._dx, s._dx, s._dx])[1]
        if zmax>(np.array(list(np.shape(s._temperature_grid)))*np.array([s._dx, s._dx, s._dx]))[2]: 
            zmax = np.array(list(np.shape(s._temperature_grid)))*np.array([s._dx, s._dx, s._dx])[2]

        # Min and Max values are all in mm. Get corresponding indices.
        x_local = np.arange(int(xmin),int(xmax),s._dx) - Tx
        y_local = np.arange(int(ymin),int(ymax),s._dx) - Ty
        z_local = np.arange(int(zmin),int(zmax),s._dx) - Tz

        # Meshgrid of distance-based locations from origin
        X, Y, Z = np.meshgrid(x_local, y_local, z_local, indexing="ij")

        xi = X + (v * (tau - s._dt))

        c = np.where(xi >= 0, c1, c2)
        f = np.where(xi >= 0, f_f, f_r)

        NewX = np.where(
            xi >= 0, 
            X + (v * (tau - s._dt)), # Deposit dir
            Y + (v * (tau - s._dt)), # Orthogonal
        )

        C_constant = (
                6 * np.sqrt(3) * f * H_i
            ) / (
                a * b * c * np.pi * np.sqrt(np.pi)
        )

        q = C_constant * np.exp(
            (-3 * (Y**2 / a**2))+(     # Orthogonal
            -3 * (Z**2 / b**2))+(     # Height
            -3 * ((NewX)**2 / c) )          # Deposit dir
        )

        # Quick validation
        # ValidQ = np.sqrt(Y**2 + xi**2) > c, q=0

        if zmin<s._substrate['thickness']:
            zmin_bead = s._substrate['thickness']

            s._bead_mask[round(int(xmin/s._dx)):round(int(xmax/s._dx)),
                        round(int(ymin/s._dx)):round(int(ymax/s._dx)),
                        round(int(zmin_bead/s._dx)):round(int(zmax/s._dx))] = 1
        else: 
            s._bead_mask[round(int(xmin/s._dx)):round(int(xmax/s._dx)),
                        round(int(ymin/s._dx)):round(int(ymax/s._dx)),
                        round(int(zmin/s._dx)):round(int(zmax/s._dx))] = 1

        local_bead_mask = s._bead_mask[round(int(xmin/s._dx)):round(int(xmax/s._dx)),
                                       round(int(ymin/s._dx)):round(int(ymax/s._dx)),
                                       round(int(zmin/s._dx)):round(int(zmax/s._dx))]
        
        pad_width = [(0, target - current) for current, target in zip(q.shape, local_bead_mask.shape)]
        q = np.pad(q, pad_width, mode='constant', constant_values=0)

        dT = (q * s._dt) / (s._rho_cp_field[round(int(xmin/s._dx)):round(int(xmax/s._dx)),
                                            round(int(ymin/s._dx)):round(int(ymax/s._dx)),
                                            round(int(zmin/s._dx)):round(int(zmax/s._dx))])

        s._temperature_grid[round(int(xmin/s._dx)):round(int(xmax/s._dx)),
                            round(int(ymin/s._dx)):round(int(ymax/s._dx)),
                            round(int(zmin/s._dx)):round(int(zmax/s._dx))] += dT
    
    def _goldak_diffuse(s): 
        k = np.zeros_like(s._temperature_grid, dtype=float)
        for num,i in enumerate(['Wire','Substrate']):
            k[s._bead_mask == num+1] = s._mat_props[i]['Thermal Conductivity']

        center = s._temperature_grid[1:-1, 1:-1, 1:-1]

        kxp = 0.5 * (k[1:-1, 1:-1, 1:-1] + k[2:, 1:-1, 1:-1])
        kxm = 0.5 * (k[1:-1, 1:-1, 1:-1] + k[:-2, 1:-1, 1:-1])
        kyp = 0.5 * (k[1:-1, 1:-1, 1:-1] + k[1:-1, 2:, 1:-1])
        kym = 0.5 * (k[1:-1, 1:-1, 1:-1] + k[1:-1, :-2, 1:-1])
        kzp = 0.5 * (k[1:-1, 1:-1, 1:-1] + k[1:-1, 1:-1, 2:])
        kzm = 0.5 * (k[1:-1, 1:-1, 1:-1] + k[1:-1, 1:-1, :-2])

        div_q = (
            (kxp * (s._temperature_grid[2:, 1:-1, 1:-1] - center) - kxm * (center - s._temperature_grid[:-2, 1:-1, 1:-1])) / s._dx**2 +
            (kyp * (s._temperature_grid[1:-1, 2:, 1:-1] - center) - kym * (center - s._temperature_grid[1:-1, :-2, 1:-1])) / s._dx**2 +
            (kzp * (s._temperature_grid[1:-1, 1:-1, 2:] - center) - kzm * (center - s._temperature_grid[1:-1, 1:-1, :-2])) / s._dx**2
        )

        s._temperature_grid[1:-1, 1:-1, 1:-1] += s._dt * div_q / s._rho_cp_field[1:-1, 1:-1, 1:-1]

        s._temperature_grid[0, :, :] = s._temperature_grid[1, :, :]
        s._temperature_grid[-1, :, :] = s._temperature_grid[-2, :, :]
        s._temperature_grid[:, 0, :] = s._temperature_grid[:, 1, :]
        s._temperature_grid[:, -1, :] = s._temperature_grid[:, -2, :]
        s._temperature_grid[:, :, -1] = s._temperature_grid[:, :, -2]
        s._temperature_grid[:, :, 0] = s._temperature_grid[:, :, 1]
        
    def _goldak_cooling(s, layer, Tx, Ty):
        # s._h_conv, s._rho_cp_field 

        sigma = 5.670374419e-14   # W/mm²/K^4
        emissivity = s._get_emissivity() # Variable depending on material
        
        T_sub = np.where(s._bead_mask==2, s._temperature_grid, 0)
        T_bead = np.where(s._bead_mask==1, s._temperature_grid, 0)
        
        # Convection
        q_conv_sub = s._h_conv * (T_sub - s._T_env, 0.0)
        q_conv_bead = s._h_conv * (T_bead - s._T_env, 0.0)

        # Radiation
        q_rad_sub = np.where(s._bead_mask==2, emissivity, 0) * sigma * (T_sub**4 - s._T_env**4)
        q_rad_bead = np.where(s._bead_mask==1, emissivity, 0) * sigma * (T_bead**4 - s._T_env**4)

        # Set unwanted interior/air to zero
        q_conv = np.zeros_like(s._temperature_grid, dtype=float)
        q_rad = np.zeros_like(s._temperature_grid, dtype=float)

        # Top surface of substrate
        top_substrate_idx = int(np.argwhere(s._bead_mask[0,0,:]==2)[-1])
        q_rad[np.argwhere(s._bead_mask[:,:,top_substrate_idx]==2)] = q_rad_sub[np.argwhere(s._bead_mask[:,:,top_substrate_idx]==2)]

        # Top surface of bead 
        top_bead_idx = int(np.argwhere(s._bead_mask[0,0,:]==1)[-1])
        q_rad[np.argwhere(s._bead_mask[:,:,top_bead_idx]==1)] = q_rad_bead[np.argwhere(s._bead_mask[:,:,top_bead_idx]==1)]

        # Unaffected areas will now have a decrease of zero :]
        q_loss = q_conv + q_rad #unaffected areas will be zero
        dT = s._dt * q_loss / (s._rho_cp_field * s._dx) #unaffected areas will be zero

        # Apply cooling
        s._temperature_grid -= np.maximum(dT, s._T_env) #unaffected areas will be zero

    # ========================================================
    # ============== ROSENTHAL MODEL EQUATIONS ===============
    # ========================================================

    def _rosenthal_applyheat(s, H_i, step_length, layer, x, y, z): 
        # Deposition radius
        r = np.sqrt(step_length**2 + (s._get_wire_width(layer)**2))

        # Find z location we're manipulating
        zmin = z - s._mat_props['Wire']['Height']
        zmax = z

        # Center deposition on curr location, plus or minus
        xmax, xmin = x + np.array([1, -1]) * ((r/2))
        ymax, ymin = y + np.array([1, -1]) * ((r/2))

        # Get grid indices
        xmin = int(round(xmin/s._dx,0)); xmax = int(round(xmax/s._dx,0))
        ymin = int(round(ymin/s._dx,0)); ymax = int(round(ymax/s._dx,0))
        zmin = int(round(zmin/s._dx,0)); zmax = int(round(zmax/s._dx,0))

        s._bead_mask[xmin:xmax, ymin:ymax, zmin:zmax] = 1

        T_bead = s._T_env + (
            (H_i / (
                2.0 * np.pi * s._mat_props['Wire']['Thermal Conductivity'] * r)
            ) * np.exp(
                -(s._mat_props['Welding Settings']['Traverse Speed']) * step_length / (
                    2.0 * s._mat_props['Wire']['Thermal Diffusivity'])
                )
        )
        T_bead /= (s._get_wire_width(layer) * .6)

        s._temperature_grid[xmin:xmax, ymin:ymax, zmin:zmax] += T_bead

    def _rosenthal_diffuse(s): 
        # Alpha should be based on the temperature field such that 
        alpha_field = np.zeros_like(s._temperature_grid, dtype=float)
        for num,i in enumerate(['Wire','Substrate']):
            alpha_field[s._bead_mask == num+1] = ((s._temperature_grid * -s._mat_props[i]['Thermal Conductivity']) / (
                s._mat_props[i]['Density'] * s._mat_props[i]['Specific Heat']
            ))[s._bead_mask == num+1]

        Tpad = np.pad(s._temperature_grid, ((1,1),(1,1),(1,1)), mode="edge")
        d2x = (Tpad[2:,1:-1,1:-1] - 2*s._temperature_grid + Tpad[:-2,1:-1,1:-1]) / (s._dx**2)
        d2y = (Tpad[1:-1,2:,1:-1] - 2*s._temperature_grid + Tpad[1:-1,:-2,1:-1]) / (s._dx**2)
        d2z = (Tpad[1:-1,1:-1,2:] - 2*s._temperature_grid + Tpad[1:-1,1:-1,:-2]) / (s._dx**2)

        s._temperature_grid += alpha_field * (d2x + d2y + d2z) * s._dt

    def _rosenthal_cooling(s):
        exponential_field = np.zeros_like(s._temperature_grid, dtype=float)
        for num,i in enumerate(['Wire','Substrate']):
            exponential_field[s._bead_mask == num+1] = np.exp(-s._dt / s._mat_props[i]['Tau'])

        s._temperature_grid = s._T_env + ((s._temperature_grid - s._T_env)*exponential_field)

# ========================================================
# ========================================================
# ========================================================

def dwell_time(material, path):
    if material=='Steel': 
        if path=='wall':
            return 30
        if path=='weave':
            return 10
        if path=='square':
            return 45
        if path=='hexagon':
            return 45
    elif material=='Aluminum':
        if path=='wall':
            return 10
        if path=='weave':
            return 10
        if path=='square':
            return 20
        if path=='hexagon':
            return 20
    else:
        ValueError('Wrong material type selected.')
    """
    Dwell Time (seconds)
        Aluminum
            Bead    - 10
            Weave   - 10
            Hexagon - 20
            Square  - 20
        Steel      ---
            Bead    - 30
            Weave   - 10
            Hexagon - 45
            Square  - 45
    """

if __name__ == "__main__":
    substrate={
        "bigSubstrate":{
            'length':    203.2,  # mm
            'width':     203.2,  # mm
            'thickness': 12.7,   # mm
        },# ---          ---     ---
        "smallSubstrate":{#-     ---
            'length':    152.4,  # mm
            'width':     50.8,   # mm
            'thickness': 12.7,   # mm
        },
    }

    points={
        'wall':{
            'pts': [(12, 25.4), (139, 25.4)],
            'type': 'Reverse',
        },
        'weave':{
            'pts': [(12.0, 8.0),(30.0, 42.8),(48.0, 8.0),(66.0, 42.8),(84.0, 8.0),(102.0, 42.8),(120.0, 8.0),(138.0, 42.8)],
            'type': 'Reverse',
        },
        'square':{
            'pts': [(38.1, 38.1), (165.1, 38.1), (165.1, 165.1), (38.1, 165.1)],
            'type': 'Cycle',
        },
        'hexagon':{
            'pts': [(170.9, 101.6), (136.3, 161.6), (67.0, 161.6), (32.3, 101.6), (67.0, 41.6), (136.3, 41.6)],
            'type': 'Cycle',
        }
    }

    # Correction from CCW to CW
    points['square']['pts']=list(reversed(points['square']['pts']))
    points['hexagon']['pts']=list(reversed(points['hexagon']['pts']))
    points['hexagon']['pts']=points['hexagon']['pts'][2:]+points['hexagon']['pts'][0:2]

    # model = Model(material  =   'Aluminum',
    #               substrate =   substrate['smallSubstrate'],
    #               pts       =   points['wall'], 
    #               dwell     =   10, 
    #               model     =   'Goldak',
    #               layers    =   10,
    #               filename  =   'KSim_Goldak_Aluminum_wall')
    # model.run()

    # model = Model(material  =   'Aluminum',
    #               substrate =   substrate['smallSubstrate'],
    #               pts       =   points['wall'], 
    #               dwell     =   10, 
    #               model     =   'Rosenthal',
    #               layers    =   10,
    #               filename  =   'KSim_Rosenthal_Aluminum_wall')
    # model.run()

    # for mod in ['Rosenthal','Goldak']:
    for mater in ['Aluminum', 'Steel']:
        for i,sub in enumerate(['smallSubstrate','bigSubstrate']):
            if i==0: 
                for path in ['wall','weave']:
                    model = Model(material    =   mater,
                                    substrate =   substrate[sub],
                                    pts       =   points[path], 
                                    dwell     =   dwell_time(mater,path), 
                                    model     =   'Rosenthal',
                                    layers    =   10,
                                    filename  =   f'KSim_Rosenthal_{mater}_{path}')
                    model.run()
            else: 
                for path in ['square','hexagon']:
                    model = Model(material    =   mater,
                                    substrate =   substrate[sub],
                                    pts       =   points[path], 
                                    dwell     =   dwell_time(mater,path), 
                                    model     =   'Rosenthal',
                                    layers    =   10,
                                    filename  =   f'KSim_Rosenthal_{mater}_{path}')
                    model.run()

    # model = Model(material  =   'Aluminum', -------------------- or 'Steel'
    #               substrate =   substrate['smallSubstrate'], --- small or big
    #               pts       =   points['hexagon'], ------------- point type
    #               dwell     =   10, ---------------------------- in seconds
    #               model     =   'Rosenthal', ------------------- or 'Goldak'
    #               filename  =   'Test') ------------------------ exclude ".csv"