import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";

import { Layout, Grid, GridItem, Heading, Button } from "../components";

import ambassadors from "./ambassadors.json";

export default function Ambassadors() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <Heading level={1}>MLflow Ambassador program</Heading>
        <div className="flex flex-col gap-2 text-xl pt-16">
          <p>The call was made, and the community answered in droves.</p>
          <p>
            We're thrilled to present our MLflow Ambassadors, a select group
            distinguished by their commitment to mentoring, teaching, and most
            importantly, building AI solutions.
          </p>
          <p>
            These ambassadors will be visible at various industry meetups and
            speaking engagements, as well as through their insightful
            contributions right here in our [MLflow Blog](blog). They'll be
            sharing their invaluable expertise, experiences, and tips on
            leveraging MLflow for practical MLOps applications, as well as
            in-depth tutorials on how to leverage some of the more advanced
            features of MLflow. Representing a wide array of backgrounds and
            industries, each ambassador brings a unique perspective fueled by a
            shared enthusiasm for innovation and education in the field.
          </p>
          <p>
            Over the next few months, you'll be seeing a lot more from them both
            on our platform, at local meetups, and as speakers on behalf of the
            MLflow project.
          </p>
        </div>
      </div>

      <div className="flex flex-col max-w-4xl mx-auto">
        <Heading level={2}>MLflow Ambassadors</Heading>
        <div className="grid grid-cols-1 sm:grid-cols-2 pt-10 lg:grid-cols-3 gap-10">
          {ambassadors.flatMap((row) =>
            row.map((item) => (
              <div
                key={item.title}
                className="bg-white/5 rounded-[16px] p-4 text-center flex flex-col items-center"
              >
                <img
                  src={useBaseUrl(item.img)}
                  alt={item.title}
                  className="rounded-[16px] p-2 w-60 h-60 object-cover"
                />
                <div>
                  <h3 className="text-xl font-semibold mt-2 pt-6">
                    {item.title}
                  </h3>
                  <div className="text-base text-white/50 mt-1">
                    {item.role}, {item.company}
                  </div>
                </div>
              </div>
            )),
          )}
        </div>
      </div>
      <div className="flex flex-col gap-10 max-w-4xl mx-auto mt-10 text-xl">
        <div className="flex flex-col gap-2">
          <Heading level={1}>About the program</Heading>

          <span className="pt-10">
            The MLflow Ambassador Program exists to empower community members
            with the tools and resources needed to:
          </span>
          <ul className="list-disc list-inside">
            <li>Promote MLflow projects and technology</li>
            <li>
              Educate a local community on the MLflow mission and technical
              aspects
            </li>
            <li>Engage in MLflow community growth</li>
          </ul>
        </div>
        <div className="flex flex-col gap-2">
          <h3>Requirements</h3>
          <ul className="list-disc list-inside">
            <li>
              Actively involved in the MLflow community as a contributor,
              blogger, speaker, etc.
            </li>
            <li>
              An active leader in the MLOps / AI community with a minimum of 1
              year of experience in:
            </li>
            <ul className="list-disc list-inside">
              <li>Organizing events (virtual/in-person)</li>
              <li>Speaking at events</li>
              <li>Mentoring others</li>
              <li>Creating content (e.g., blogs, videos, etc.)</li>
            </ul>
          </ul>
        </div>
        <div className="flex flex-col gap-2">
          <h3>Responsibilities</h3>
          <ul className="list-disc list-inside">
            <li>
              Contribute technical content such as blog posts, video tutorials,
              training modules, etc.
            </li>
            <li>
              Organize and host at least one local MLflow community event/year
              (Meetup).
            </li>
            <li>Help the community learn more about MLflow</li>
            <li>
              Advocate for MLflow at events, evangelizing and disseminating
              information about MLflow.
            </li>
            <li>
              Be a source of information and support for those interested in
              MLflow and help the local community learn more about MLflow.
            </li>
            <li>
              Facilitate the local community's understanding and exploration of
              MLflow.
            </li>
            <li>
              Publicly represent and uphold the interests of the MLflow
              community.
            </li>
          </ul>
        </div>

        <div>
          <p>
            The MLflow Ambassador Selection Committee reviews applications on a
            rolling basis. We are focused on creating a group of Ambassadors
            that meet all our requirements and represent our community and
            geographical diversity. Once you submit your application, you enter
            the pool of applicants that get reviewed on a rolling basis.{" "}
          </p>
          <p>
            Successful ambassadors encompass engineers, developers, bloggers,
            influencers, and evangelists who are actively involved in MLflow.
            They contribute to work groups, online communities, community
            events, training sessions, workshops, and various related
            activities.
          </p>
          <p>Are you interested in becoming an official MLflow Ambassador?</p>
          <div className="pt-4 flex justify-center">
            <Link href="https://forms.gle/foW9ZtietYLLYCp99">
              <Button variant="primary" size="small">
                Apply
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </Layout>
  );
}
