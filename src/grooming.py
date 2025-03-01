from arg_scripts.grooming_args import GroomingProps


class Grooming:
    """
    This class contains methods related to traffic grooming.
    """
    def __init__(self, engine_props: dict, sdn_props: object):
        self.grooming_props = GroomingProps()
        self.engine_props = engine_props
        self.sdn_props = sdn_props

    def _end_to_end_grooming(self):
        """
        Function to groom the arrival requests to already established lightpaths.

        :return: If the requested is groomed to established lightpath.
        :rtype: bool
        """
                
        light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))

        if light_id in self.sdn_props.lightpath_status_dict:
            path_groups = {}
            for lp_id, lp_info in self.sdn_props.lightpath_status_dict[light_id].items():
                if lp_info["remaining_bandwidth"] > 0:
                    path_key = tuple(lp_info["path"])
                    reverse_path_key = tuple(reversed(lp_info["path"]))

                    # Normalize the path key to treat paths and their reverses as the same group
                    normalized_path_key = min(path_key, reverse_path_key)
                    if normalized_path_key not in path_groups:
                        path_groups[normalized_path_key] = {
                            "total_remaining_bandwidth": 0,
                            "lightpaths": [],
                            "lp_id_list": []
                        }
                    path_groups[normalized_path_key]["total_remaining_bandwidth"] += lp_info["remaining_bandwidth"]
                    path_groups[normalized_path_key]["lightpaths"].append((lp_id, lp_info))
                    path_groups[normalized_path_key]["lp_id_list"].append(lp_id)

            # Find the path group with the maximum total remaining bandwidth
            max_path_group = max(path_groups.values(), key=lambda group: group["total_remaining_bandwidth"], default=None)
            if max_path_group and max_path_group["total_remaining_bandwidth"] != 0:
                remaining_bw = int(self.sdn_props.bandwidth)
                for lp_id in max_path_group['lp_id_list']:
                    if self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"] == 0:
                        continue
                    if self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"] > remaining_bw :
                        tmp_remaining_bw = remaining_bw
                        remaining_bw = 0
                    else:
                        tmp_remaining_bw = self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"]
                        remaining_bw -= self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"]
                        self.sdn_props.is_sliced = True
                    self.sdn_props.lightpath_status_dict[light_id][lp_id]["requests_dict"].update({self.sdn_props.req_id:tmp_remaining_bw})
                    self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"] -= tmp_remaining_bw
                    lp_usage = (1-(self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"]/self.sdn_props.lightpath_status_dict[light_id][lp_id]['lightpath_bandwidth']))
                    self.sdn_props.lightpath_status_dict[light_id][lp_id]["time_bw_usage"].update({self.sdn_props.arrive: lp_usage * 100})
                    self.sdn_props.bandwidth_list.append(str(tmp_remaining_bw))
                    self.sdn_props.core_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]["core"])
                    self.sdn_props.band_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]["band"])
                    self.sdn_props.start_slot_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]["start_slot"])
                    self.sdn_props.end_slot_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]["end_slot"])
                    self.sdn_props.modulation_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]["mod_format"])
                    self.sdn_props.path_list = self.sdn_props.lightpath_status_dict[light_id][lp_id]["path"]
                    self.sdn_props.xt_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]["snr_cost"])
                    self.sdn_props.lightpath_bandwidth_list.append(self.sdn_props.lightpath_status_dict[light_id][lp_id]['lightpath_bandwidth'])
                    self.sdn_props.lightpath_id_list.append(lp_id)
                    if remaining_bw == 0:
                        self.sdn_props.was_routed = True
                        self.sdn_props.was_groomed = True
                        self.sdn_props.was_partially_groomed = False
                        self.sdn_props.num_trans = 0
                        self.sdn_props.was_new_lp_established = list()
                        self.sdn_props.remaining_bw = "0"
                        return True
                self.sdn_props.was_partially_groomed = True
                self.sdn_props.was_groomed = False
                self.sdn_props.remaining_bw = remaining_bw
        return False
    def _release_service(self):
        """
        Removes a previously allocated request from the lightpaths.
        
		:return: list of the lightpaths which are not carried any request.
        :rtype: list
        """

        release_lp = []
        light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))

        for lp_id in self.sdn_props.lightpath_id_list[:]: 
            index = self.sdn_props.lightpath_id_list.index(lp_id)
            req_bw = self.sdn_props.lightpath_status_dict[light_id][lp_id]['requests_dict'][self.sdn_props.req_id]
            self.sdn_props.lightpath_status_dict[light_id][lp_id]['requests_dict'].pop(self.sdn_props.req_id)
            self.sdn_props.lightpath_status_dict[light_id][lp_id]['remaining_bandwidth'] += req_bw
            self.sdn_props.remaining_bw = int(self.sdn_props.remaining_bw) - req_bw
            self.sdn_props.lightpath_id_list.pop(index)
            self.sdn_props.lightpath_bandwidth_list.pop(index)
            if self.sdn_props.lightpath_status_dict[light_id][lp_id]['remaining_bandwidth'] == self.sdn_props.lightpath_status_dict[light_id][lp_id]['lightpath_bandwidth'] :
                release_lp.append(lp_id)
            else:
                lp_usage = (1-(self.sdn_props.lightpath_status_dict[light_id][lp_id]["remaining_bandwidth"]/self.sdn_props.lightpath_status_dict[light_id][lp_id]['lightpath_bandwidth']))
                self.sdn_props.lightpath_status_dict[light_id][lp_id]["time_bw_usage"].update({self.sdn_props.depart: lp_usage * 100})
        return release_lp

    def handle_grooming(self, request_type):
        """
        Controls the methods of this class.
        """

        if request_type == "release":
            return self._release_service()
        else:
            return self._end_to_end_grooming()
