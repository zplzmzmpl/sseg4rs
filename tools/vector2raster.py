# -*- coding: utf-8 -*-
import os
import math
from osgeo import gdal, ogr
import argparse

# 类别到整数值的映射字典
VALUE_MAP = {
    "草地": 1,
    "建设用地": 2,
    "林地": 3,
    "湾内水域": 4,
    "园地": 5,
    "耕地": 6,
    "河流": 7,
    "红树林": 8,
    "裸地": 9,
    "滩涂": 10,
    "水库坑塘": 11,
    "养殖": 12
}

def vector_to_raster(input_folder, output_folder, pixel_size=30):
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.shp'):
                shp_path = os.path.join(root, file)
                process_shapefile(shp_path, output_folder, pixel_size)

def process_shapefile(shp_path, output_folder, pixel_size):
    src_ds = ogr.Open(shp_path)
    if not src_ds:
        print(f"unable to open file: {shp_path}")
        return

    # 确保输出目录存在
    src_layer = src_ds.GetLayer()
    srs = src_layer.GetSpatialRef()
    x_min, x_max, y_min, y_max = src_layer.GetExtent()

    # 计算栅格尺寸
    cols = math.ceil((x_max - x_min) / pixel_size)
    rows = math.ceil((y_max - y_min) / pixel_size)

    # 创建输出路径
    base_name = os.path.splitext(os.path.basename(shp_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_raster.tif")

    # 创建输出栅格
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    if srs:
        out_ds.SetProjection(srs.ExportToWkt())
    
    # 初始化
    raster_band = out_ds.GetRasterBand(1)
    raster_band.Fill(0)

    # 创建内存临时图层
    mem_ds, mem_layer = create_memory_layer(src_layer)

    # 转换属性字段
    transform_attributes(src_layer, mem_layer)

    # 执行栅格化
    gdal.RasterizeLayer(out_ds, [1], mem_layer, options=["ATTRIBUTE=CODE"])

    # 清理资源
    out_ds.FlushCache()

    out_ds = None
    src_ds = None
    mem_ds = None
    print(f"successfully handle {output_path}")

def create_memory_layer(src_layer):
    """创建内存临时图层"""
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('temp')
    mem_layer = mem_ds.CreateLayer('temp', 
                                 srs=src_layer.GetSpatialRef(),
                                 geom_type=src_layer.GetGeomType())
    
    # 复制字段结构
    src_defn = src_layer.GetLayerDefn()
    for i in range(src_defn.GetFieldCount()):
        mem_layer.CreateField(src_defn.GetFieldDefn(i))
    
    # 添加CODE字段
    code_field = ogr.FieldDefn('CODE', ogr.OFTInteger)
    mem_layer.CreateField(code_field)
    return mem_ds, mem_layer

def transform_attributes(src_layer, mem_layer):
    """转换属性字段到数值编码"""
    for src_feat in src_layer:
        geom = src_feat.GetGeometryRef()

        # 跳过空几何对象
        if geom is None or geom.IsEmpty():
            continue

        geom = geom.Clone()
        
        # 检查几何对象类型 (wkbPolygon=3, wkbMultiPolygon=6)
        geom_type = geom.GetGeometryType()
        lucc_value = src_feat.GetField('LUCC15')
        
        # 创建新要素
        mem_feat = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feat.SetGeometry(geom)
        
        for i in range(src_feat.GetFieldCount()):
            mem_feat.SetField(i, src_feat.GetField(i))
        
        # 设置CODE
        if geom_type == ogr.wkbPolygon or geom_type == ogr.wkbMultiPolygon:
            code = VALUE_MAP.get(lucc_value, 255)  # Use specified values for valid polygons
        else:
            code = 0  # Set non-polygon features to 0 (invalid areas)
        
        mem_feat.SetField('CODE', code)
        mem_layer.CreateFeature(mem_feat)
        mem_feat = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert vector data to raster")
    parser.add_argument("--input_folder", type=str, help="Path to the folder containing shapefiles")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    args = parser.parse_args()
    # input_folder = r"E:\2025\lucc20181114\15lucc-20181114"
    # output_folder = r"D:\FinalDesignData\LUCC15_raster"
    
    vector_to_raster(args.input_folder, args.output_folder)
    print("Done!")