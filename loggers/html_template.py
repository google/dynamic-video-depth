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

TABLE_HEADER = """
<html>
<head>
    <script type="text/javascript" language="javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/unveil/1.3.0/jquery.unveil.min.js" integrity="sha512-smKadbDZ1g5bsWtP1BuWxgBq1WeP3Se1DLxeeBB+4Wf/HExJsJ3OV6lzravxS0tFd43Tp4x+zlT6/yDTtr+mew==" crossorigin="anonymous"></script>
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
                "drawCallback": function( settings ) {{
            $("#myTable img:visible").unveil();
        }},

            }});
        }});
    </script>
</head>

<body bgcolor='black'>
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
</body>
</html>
"""
image_tag_template = "<td><img src=\"{image_path}\" style=\"max-width:100%;height:auto;\"></td>"
