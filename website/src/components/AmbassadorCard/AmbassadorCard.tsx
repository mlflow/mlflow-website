import LearnCard from "../LearnCard/LearnCard";

interface AmbassadorCardProps {
  title: string;
  role: string;
  company: string;
  companyLink: string;
  personalLink: string;
  img: string;
}

export const AmbassadorCard = ({
  title,
  role,
  company,
  companyLink,
  personalLink,
  img,
}: AmbassadorCardProps) => (
  <LearnCard
    title={title}
    content={
      <span>
        {role} <br />
        <a href={companyLink} target="_blank" rel="noopener noreferrer">
          {company}
        </a>
      </span>
    }
    href={personalLink}
    img={img}
  />
);
