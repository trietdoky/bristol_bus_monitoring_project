from BODSDataExtractor.extractor import TimetableExtractor
import pandas as pd
import numpy as np
import requests
from lxml import etree as et
# from xml.etree import ElementTree as et
from time import sleep
from datetime import datetime, timedelta
import xml.dom.minidom
import xmltodict
import json
from bs4 import BeautifulSoup as bsoup
import time
import schedule
import os
import shutil
import pytz
import re
import geopy.distance
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from pyproj import Geod
import uuid
import hashlib