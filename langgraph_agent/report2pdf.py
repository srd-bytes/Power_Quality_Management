import markdown2  # Markdown → HTML conversion
from weasyprint import HTML  # HTML → PDF rendering engine
from pathlib import Path



def get_report(md_string, path):
    """Convert a markdown report string into a PDF saved under the given path."""
    html = markdown2.markdown(md_string, extras=["tables", "fenced-code-blocks"])
    css = """
    <html><head><style>
    body { font-family: Arial; font-size: 10pt; margin: 0.5cm; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #aaa; padding: 4px; text-align: left; }
    h1, h2, h3 { margin: 6px 0; }
    img { max-width: 250px; max_height:125px ;height: auto; }
    </style></head><body>
    """ + html + "</body></html>"

    HTML(string=css).write_pdf(f"{path}/power_quality_report.pdf")

# def get_report(md_string,path):
#     html = markdown2.markdown(md_string, extras=["tables", "fenced-code-blocks"])

#     compact_css = """
#     <html>
#     <head>
#         <style>
#             @page {
#                 size: A4;
#                 margin: 1cm;
#             }
#             body {
#                 font-family: 'Arial', sans-serif;
#                 font-size: 10px;
#                 padding: 0;
#                 line-height: 1.2;
#             }
#             h1 { font-size: 14px; margin: 4px 0; }
#             h2 { font-size: 12px; margin: 3px 0; }
#             h3 { font-size: 11px; margin: 2px 0; }
#             table {
#                 width: 100%;
#                 border-collapse: collapse;
#                 margin: 5px 0;
#             }
#             th, td {
#                 border: 1px solid #ccc;
#                 padding: 3px 6px;
#             }
#             th {
#                 background-color: #eee;
#             }
#             ul, ol {
#                 margin: 0 0 4px 15px;
#             }
#             p {
#                 margin: 3px 0;
#             }
#             code {
#                 background-color: #f4f4f4;
#                 padding: 1px 3px;
#                 font-size: 9px;
#             }
#         </style>
#     </head>
#     <body>
#     """ + html + "</body></html>"

#     HTML(string=compact_css).write_pdf(f"{path}/compact_power_quality_report.pdf")

