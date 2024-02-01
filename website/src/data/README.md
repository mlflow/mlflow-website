# Adding companies to the "Powered By" page

Follow these steps to add a company logo to the "Powered By" page:

1. Edit the `companies.json` file. The key should be the company's display name (it's used for
   alt/hover text on the image), and the value should be the company's website URL.
2. Add the SVG file to `/static/img/companies`. The filename must be the company's display name
   in lowercase, with all spaces replaced with `-`. For example, `"University of Washington" -> university-of-washington.svg`.
