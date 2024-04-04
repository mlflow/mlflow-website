import LearnCard from "../LearnCard";

export const createAmbassadorCard = ({
  title,
  role,
  company,
  companyLink,
  personalLink,
  img,
}) => (
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
