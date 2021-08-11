# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join, dirname, basename
from glob import glob
from os import makedirs


class Webpage():
    WEB_TEMPLATE = """
    <html>
    <head>
    <script type="text/javascript" language="javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
    <style>
        .plotly-graph-div{{
        margin:auto;
    }}
    </style>
    <script type="text/javascript" language="javascript"
        src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" language="javascript"
        src="https://cdn.datatables.net/buttons/1.6.1/js/dataTables.buttons.min.js"></script>
    <script type="text/javascript" language="javascript"
        src="https://cdn.datatables.net/buttons/1.6.1/js/buttons.colVis.min.js"></script>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css">
    </link>
    <link rel="stylesheet" type="text/css"
        href="https://ztzhang.info/assets/css/buttons.dataTables.min.css">
    </link>

    <link rel="stylesheet" type="text/css"
        href="https://ztzhang.info/assets/css/datatable.css">
    </link>
    <script>
        $(document).ready(function () {{
            var table = $('#myTable').DataTable({{
                dom: 'Blfrtip',
                autoWidth: false,
                buttons: [
                    'columnsToggle'
                ],
                "lengthMenu": [[5, 10, 15, 20, -1], [5, 10, 15, 20, "All"]],
                "columnDefs": [
                    {{"targets": "_all",
                    "className": "dt-center"}}
                ],
            }});
        }});
    </script>
    </head>

    <body bgcolor='black'>
        {body_content}
    </body>
    </html>
"""
    image_tag_template = "<td><img src=\"{image_path}\" style=\"max-width:100%;height:auto;\"></td>"
    table_template = """
        <table id="myTable" class="cell-border compact stripe">
            <thead>
                <tr>
                    {table_header}
                </tr>
            </thead>
            <tbody>

                    {table_body}
            </tbody>
        </table>
    """

    def __init__(self, notable=False):
        self.content = self.WEB_TEMPLATE
        self.table_content = self.table_template
        self.video_content = ''
        if not notable:
            self.devider = f'<hr><div style="text-align:center; font-size:20px;color:ffffff;">data table</div><br>'
        else:
            self.devider = ''

    def add_image_table_from_folder(self, path, img_prefixes, keys=None, rel_path='./'):
        if keys is None:
            keys = img_prefixes
        header = ''
        for k in keys:
            header += f"<th>{k}</th>\n"
        content = ""
        file_lists = {}

        for prefix in img_prefixes:
            file_lists[prefix] = sorted(glob(join(path, prefix + '*')))
            l = len(file_lists[prefix])
        for i in range(l):
            content += "<tr>\n"
            for k in file_lists.keys():
                link = join(rel_path, basename(file_lists[k][i]))
                content += f"<td><img src=\"{link}\" style=\"max-width:100%;height:auto;\"></td>\n"
            content += "</tr>\n"
        self.table_content = self.table_content.format(table_header=header, table_body=content)

    def add_video(self, rel_path_to_video, title=''):
        video_tag = f'<div style="text-align:center; font-size:20px;color:ffffff;">{title}<br><video width="40%" max-width="40%" height="auto" autoplay loop laysinline muted > <source src="{rel_path_to_video}" type="video/mp4"> </video><br><br></div>'
        self.video_content += video_tag

    def add_div(self, div_string):
        self.video_content += div_string

    def save(self, path):
        content = self.content.format(body_content=self.video_content + self.devider + self.table_content)
        d = dirname(path)
        makedirs(d, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return
