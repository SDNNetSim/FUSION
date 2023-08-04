import math

import numpy as np



class SnrMeasurments:
    """
    Calculates SNR for a given request.
    """
    
    def __init__(self, path, modulation_format, start_slot_no, end_slot_no, no_assigned_slots, 
                 assigned_core_no, requested_bit_rate, physical_topology, frequncy_spacing, input_power, 
                 spectral_slots, SNR_requested, 
                 requests_status = None, phi = None, network_spec_db=None, 
                 guard_band=0, baud_rates = None, EGN = None,  bidirectional = True):
        self.path = path

        self.no_assigned_slots = no_assigned_slots
        self.start_slot_no = start_slot_no
        self.end_slot_no = end_slot_no
        self.assigned_core_no = assigned_core_no
        self.requested_bit_rate = requested_bit_rate
        self.modulation_format = modulation_format
        self.guard_band = guard_band
        self.physical_topology = physical_topology
        self.network_spec_db = network_spec_db
        self.frequncy_spacing = frequncy_spacing
        self.input_power = input_power
        self.spectral_slots = spectral_slots
        self.SNR_requested = SNR_requested
        self.phi = phi
        self.EGN = EGN
        self.requests_status = requests_status
        self.baud_rates = baud_rates
        self.bidirectional = bidirectional
        # self.mode_coupling_co = mode_coupling_co
        # self.bending_radius = bending_radius
        # self.propagation_const = propagation_const
        # self.core_pitch = core_pitch
        
        self.cores_matrix = None
        self.rev_cores_matrix = None
        self.num_slots = None

        self.response = {'SNR': None }
        
        
    def G_NLI_ASE(self):
        light_frequncy = (1.9341 * 10 ** 14)
        
        Fi = ((self.start_slot_no * self.frequncy_spacing ) + ( ( self.no_assigned_slots * self.frequncy_spacing ) / 2 )) * 10 ** 9
        BW = self.no_assigned_slots * self.frequncy_spacing * 10 ** 9
        PSDi = self.input_power / BW
        
        PSD_NLI = 0
        PSD_corr = 0
        for link in range(0, len(self.path)-1):
            MCI = 0
            Num_span = 0
            visited_channel = []
            link_id = self.network_spec_db[(self.path[link], self.path[link+1])]['link_num']
            Rho = ( ( math.pi ** 2 ) * np.abs( self.physical_topology['links'][link_id]['fiber']['dispersion'] ) )/( 2 * self.physical_topology['links'][link_id]['fiber']['attenuation'])
            Mio = ( 3 * ( self.physical_topology['links'][link_id]['fiber']['non_linearity'] ** 2 ) ) / ( 2 * math.pi * self.physical_topology['links'][link_id]['fiber']['attenuation'] * np.abs( self.physical_topology['links'][link_id]['fiber']['dispersion'] ))
            SCI = (PSDi ** 2) * math.asinh( Rho * (BW ** 2 ) )
            for w in range(self.spectral_slots):
                if self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.assigned_core_no][w] > 0: #!= 0 :
                    if self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.assigned_core_no][w] in visited_channel:
                        continue
                    else:
                        visited_channel.append(self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.assigned_core_no][w])
                    BW_J = len(np.where(self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.assigned_core_no][w] == self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.assigned_core_no])[0]) * self.frequncy_spacing    
                    Fj = (( w * self.frequncy_spacing)+((BW_J ) / 2 ) )* 10 ** 9
                    BWj = BW_J * 10 **9
                    PSDj = self.input_power / BWj
                    if Fi != Fj:
                        MCI = MCI + ((PSDj ** 2) * math.log( abs((abs(Fi-Fj)+(BWj/2))/(abs(Fi-Fj)-(BWj/2)))))
            
            if self.phi:
                hn = 0
                for i in range(1,math.ceil( ( len(visited_channel) - 1 ) / 2 )+1):
                    hn = hn + 1 / i
                effective_L = ( 1 - math.e ** ( -2 * self.physical_topology['links'][link_id]['fiber']['attenuation'] * self.physical_topology['links'][link_id]['fiber']['span_length'] * 10**3) ) / ( 2 * self.physical_topology['links'][link_id]['fiber']['attenuation'])
                baud_rate = int(self.requested_bit_rate) *10**9 / 2 #self.baud_rates[self.modulation_format]
                temp_coef = ((self.physical_topology['links'][link_id]['fiber']['non_linearity'] ** 2 ) * (effective_L ** 2) * (PSDi ** 3) *  ( BW ** 2 ) ) / ( ( baud_rate ** 2 ) * math.pi * self.physical_topology['links'][link_id]['fiber']['dispersion'] * (self.physical_topology['links'][link_id]['fiber']['span_length']*10**3))
                PSD_corr = ( 80 / 81 ) * self.phi[self.modulation_format] * temp_coef * hn
            
            
            PSD_ASE = 0
            if self.EGN:
                PSD_NLI = ( ( ( SCI + MCI ) * Mio * PSDi) ) - PSD_corr
            else:
                PSD_NLI = ( ( ( SCI + MCI ) * Mio * PSDi) )
            PSD_ASE = ( self.physical_topology['links'][link_id]['fiber']['plank'] * light_frequncy * self.physical_topology['links'][link_id]['fiber']['nsp'] ) * ( math.e ** (self.physical_topology['links'][link_id]['fiber']['attenuation'] * self.physical_topology['links'][link_id]['fiber']['span_length'] * 10 ** 3 ) - 1 )
            
            
            XT_lambda = (2 * self.physical_topology['links'][link_id]['fiber']["bending_radius"] * self.physical_topology['links'][link_id]['fiber']["mode_coupling_co"] ** 2) / (self.physical_topology['links'][link_id]['fiber']["propagation_const"] * self.physical_topology['links'][link_id]['fiber']["core_pitch"])
            no_adjacent_core = 6
            lng = self.physical_topology['links'][link_id]['fiber']['span_length']*10*10**3
            XT_calc = (no_adjacent_core * ( 1 - math.exp(-(no_adjacent_core+1)*2*XT_lambda*lng))) / (1+no_adjacent_core * math.exp(-(no_adjacent_core+1)*2*XT_lambda*lng))
            P_XT = no_adjacent_core * XT_lambda * self.physical_topology['links'][link_id]['fiber']['span_length'] * 10**3 * self.input_power 
            #for i in range(1,100):
            #P_XT2 = self.input_power * math.exp(-)
            # SNR =( 1 / ( PSDi / ( ( PSD_ASE + PSD_NLI ) * Num_span ) ) )
            for i in range(1,100):
                Num_span =  i
                lng2 = Num_span * self.physical_topology['links'][link_id]['fiber']['span_length'] 
                P_XT2 = no_adjacent_core * XT_lambda * self.input_power * math.exp(-self.physical_topology['links'][link_id]['fiber']['attenuation'] * lng2) * lng2 * 10**3
                P_XT2 = P_XT2 * self.no_assigned_slots
                SNR = ( 1 / ( PSDi*BW / ( ( PSD_ASE*BW + PSD_NLI*BW ) * Num_span + P_XT2 ) ) )
                SNR2 = 10*math.log10(1/SNR) 
                if self.modulation_format == '64-QAM':
                    snr_tr = 22#13.5
                elif self.modulation_format == '16-QAM':
                    snr_tr = 16 #9.5
                elif self.modulation_format == 'QPSK':
                    snr_tr = 7.5
                if SNR2 < snr_tr:
                    print( "Maximum distance:  " , (i-1) * self.physical_topology['links'][link_id]['fiber']['span_length'] )
                    break
                


            
            