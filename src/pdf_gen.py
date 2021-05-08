#!/usr/bin/env python
import os

from fpdf import FPDF, HTMLMixin
from PIL import Image

# wrapper(w, h=0, txt='', border=0, ln=0, align='', fill=0, link='')


class PDF(FPDF, HTMLMixin):
    def header(self):
        # Logo
        # self.image('logo_pb.png', 10, 8, 33)
        # Arial bold 15
        # Title
        self.set_font('Arial', 'B', 16)
        self.image(os.path.join('assets', 'simple-logo-azul.png'), 10, 12, 33)
        self.cell(190, 8, 'SIMPLE - HIPAE', 0, 1, 'C')

        # Subtitle
        self.set_font('Arial', '', 14)
        self.cell(190, 7, 'Reporte de variables', 0, 0, 'C')
        # Line break
        self.ln(15)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Página ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

# Instantiation of inherited class
