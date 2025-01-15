from .points import PointCloud

from .points import POINTSCLOUD_REGISTRY

def parse_point_cloud(cfg, datapipeline):
    
    if len(cfg) == 0:
        return None
    point_cloud_type = cfg.point_cloud_type
    point_cloud = POINTSCLOUD_REGISTRY.get(point_cloud_type)
    assert point_cloud is not None, "Point Cloud is not registered: {}".format(
        point_cloud_type
    )
    init_point_cloud_path = cfg.pop("init_ply", None)
    if init_point_cloud_path:
        datapipeline.reload_point_cloud(init_point_cloud_path, sample_num=10000)
    return point_cloud(cfg, datapipeline.point_cloud)